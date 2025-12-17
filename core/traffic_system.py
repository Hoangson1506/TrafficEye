import torch
import numpy as np
import supervision as sv
from collections import deque
import queue
import threading
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

from track.sort import SORT
from track.bytetrack import ByteTrack
from detect.detect import inference_video
from core.vehicle import Vehicle
from core.violation import RedLightViolation
from core.violation_manager import ViolationManager
from core.license_plate_recognizer import LicensePlateRecognizer
from utils.config import load_config
from utils.io import violation_save_worker
from detect.utils import preprocess_detection_result
from utils.zones import load_zones
from utils.drawing import render_frame
from core.light_signal_detector import LightSignalDetector
from core.light_signal_FSM import LightSignalFSM

class TrafficSystem:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # Load models
        print("Loading models...")
        self.device = self.config.get('system', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        
        self.vehicle_model_path = self.config.get('system', {}).get('vehicle_model', "detect_gtvn.pt")
        self.license_model_path = self.config.get('system', {}).get('license_model', "lp_yolo11s.pt")
        # self.character_model_path = self.config.get('system', {}).get('character_model', "yolo11s.pt") # Unused?
        
        self.vehicle_model = YOLO(self.vehicle_model_path, task='detect', verbose=False)
        self.license_model = YOLO(self.license_model_path, task='detect', verbose=False)
        self.character_model = PaddleOCR(use_angle_cls=True, lang='en')
        
        self.tracker_instance = None
        self.violation_manager = None
        self.polygon_zone = None
        self.violation_queue = queue.Queue()
        self.worker_thread = None
        
        self.running = False
        self.generator = None
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        
        # State
        self.data_path = self.config.get('system', {}).get('data_path', "data/traffic_video.avi") 
        self.tracker_name = self.config.get('system', {}).get('tracker', "bytetrack")
        
        # Initialize worker
        self.start_worker()

        # First frame for drawing zones
        self.first_frame = None

    def start_worker(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=violation_save_worker, args=(self.violation_queue,), daemon=True)
            self.worker_thread.start()

    def update_config(self, new_config):
        # Update internal config dict
        self.config = new_config
        
    def set_source(self, data_path, tracker_name):
        self.data_path = data_path
        self.tracker_name = tracker_name
        
    def init_tracker(self):
        conf_threshold = 0.25
        if self.tracker_name == 'sort':
            cfg = self.config['tracking']['sort']
            self.tracker_instance = SORT(
                cost_function=cfg['cost_function'], 
                max_age=cfg['max_age'], 
                min_hits=cfg['min_hits'], 
                iou_threshold=cfg['iou_threshold'], 
                tracker_class=Vehicle
            )
            conf_threshold = cfg['conf_threshold']
        elif self.tracker_name == 'bytetrack':
            cfg = self.config['tracking']['bytetrack']
            self.tracker_instance = ByteTrack(
                cost_function=cfg['cost_function'], 
                max_age=cfg['max_age'], 
                min_hits=cfg['min_hits'], 
                high_conf_threshold=cfg['high_conf_threshold'], 
                low_conf_threshold=cfg['low_conf_threshold'],
                high_conf_iou_threshold=cfg['high_conf_iou_threshold'],
                low_conf_iou_threshold=cfg['low_conf_iou_threshold'],
                tracker_class=Vehicle
            )
            conf_threshold = cfg['conf_threshold']
        
        return conf_threshold

    def capture_first_frame(self):
        """Capture a single frame from the source for setting up zones."""
        if self.first_frame is None:
            return None
        return cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)

    def start(self):
        self.running = True
        # self.generator = self._process_flow() # Removed to avoid double stream initialization

    def stop(self):
        self.running = False
        self.generator = None

    def filter_vehicles_in_zone(self, tracked_objs, all_tracked_objs, sv_detections, frame_counter=0, buffer_maxlen=5):
        # Trigger zones
        in_zone_mask = self.polygon_zone.trigger(detections=sv_detections)

        for obj in all_tracked_objs:
            if obj.is_being_tracked == False and sv_detections.tracker_id is not None and obj.id in sv_detections.tracker_id[in_zone_mask]:
                obj.is_being_tracked = True
            if obj.bboxes_buffer is not None:
                obj.bboxes_buffer.append((frame_counter, obj.get_state()[0]))
            else:
                obj.bboxes_buffer = deque(maxlen=buffer_maxlen)
                obj.bboxes_buffer.append((frame_counter, obj.get_state()[0]))

        visualized_tracked_objs = [obj for obj in tracked_objs if obj.is_being_tracked]
        visualize_mask = np.isin(sv_detections.tracker_id, [obj.id for obj in visualized_tracked_objs])
        visualized_sv_detections = sv_detections[visualize_mask]

        return visualized_tracked_objs, visualized_sv_detections

    def _init_light_detector(self, h, w, light_zones_config):
        """
        Initialize LightSignalDetector from saved zone configuration.
        Returns None if no zones are configured.
        """
        # Check if any zones are configured
        has_zones = False
        for direction in ['straight', 'left', 'right']:
            if light_zones_config.get(direction, []):
                has_zones = True
                break
        
        if not has_zones:
            return None
        
        # Create detector without interactive drawing
        detector = LightSignalDetector.__new__(LightSignalDetector)
        detector.straight_light_zones = []
        detector.left_light_zones = []
        detector.right_light_zones = []
        detector.zone_masks = {'straight': [], 'left': [], 'right': []}
        
        # Load zones from config (points are stored as [top_left, bottom_right] pairs)
        for direction in ['straight', 'left', 'right']:
            points = light_zones_config.get(direction, [])
            zone_list = getattr(detector, f'{direction}_light_zones')
            
            for i in range(0, len(points), 2):
                if i + 1 < len(points):
                    top_left = points[i]
                    bottom_right = points[i + 1]
                    # Convert 2 corner points to 4-point polygon
                    polygon = [
                        (top_left[0], top_left[1]),
                        (bottom_right[0], top_left[1]),
                        (bottom_right[0], bottom_right[1]),
                        (top_left[0], bottom_right[1])
                    ]
                    zone_list.append(polygon)
        
        # Build masks
        detector.build_zone_mask(h, w)
        
        return detector

    def _process_flow(self):
        # Setup source
        source_path = self.data_path
        if source_path == "cam_ai":
            source_path = "rtsp://localhost:8554/cam_ai"
        
        conf_threshold = self.init_tracker()
        
        # Inference generator
        dets = inference_video(
            model=self.vehicle_model,
            data_path=source_path,
            output_path=None,
            device=self.device,
            stream=True,
            conf_threshold=conf_threshold,
            classes=self.config['detections']['classes'],
            imgsz=self.config['detections']['imgsz'],
            iou_threshold=self.config['detections']['iou_threshold'],
            stream_buffer=False
        )

        first_run = True
        FPS = 30
        frame_buffer = None
        
        for result in dets:
            if not self.running:
                break
                
            if first_run:
                self.first_frame = result.orig_img
                FPS = self.config['violation']['fps'] if self.config['violation']['fps'] is not None else 30
                
                # Load zones 
                zones = load_zones()
                polygon_points = zones.get("polygon", [])
                lines_config = zones.get("lines_config", {}) # Expecting a dict of categories now
                # Backward compatibility or fallback if 'lines' exists as a flat list
                if "lines" in zones and not lines_config:
                     # Default to violation_lines
                     lines_config["violation_lines"] = zones["lines"]
                
                # Default polygon if none
                if len(polygon_points) < 3:
                     # Fallback to full frame or center?
                     # Let's just default to a small box if missing
                     h, w = self.first_frame.shape[:2]
                     polygon_points = [[w//4, h//4], [w*3//4, h//4], [w*3//4, h*3//4], [w//4, h*3//4]]

                polygon_points = np.array(polygon_points, dtype=int)
                self.polygon_zone = sv.PolygonZone(polygon_points, triggering_anchors=[sv.Position.CENTER])
                
                # Frame buffer
                buffer_duration = self.config['violation']['video_proof_duration']
                buffer_maxlen = int(FPS * buffer_duration)
                frame_buffer = deque(maxlen=buffer_maxlen)
                
                # Initialize Violation Manager
                violations = [RedLightViolation(polygon_points=polygon_points, lines=lines_config, frame=self.first_frame, window_name="Traffic Violation")]
                licensePlate_recognizer = LicensePlateRecognizer(license_model=self.license_model, character_model=self.character_model)
                self.violation_manager = ViolationManager(violations=violations, recognizer=licensePlate_recognizer)
                
                # Initialize Light Signal Detector from saved zones
                light_zones_config = zones.get("light_zones", {})
                h, w = self.first_frame.shape[:2]
                light_detector = self._init_light_detector(h, w, light_zones_config)
                light_fsm = None
                if light_detector is not None:
                    initial_light_list = light_detector.detect_light_signals(self.first_frame)
                    processed_initial_lights = []
                    for light in initial_light_list:
                        if light is None:
                            processed_initial_lights.append(light)
                        else:
                            processed_initial_lights.append(light[0])  # Extract only the state
                    light_fsm = LightSignalFSM(initial_states=processed_initial_lights)
                
                frame_counter = 0
                first_run = False
            
            # Preprocess
            frame, det = preprocess_detection_result(result)
            frame_counter += 1
            
            # Tracking
            tracked_objs = self.tracker_instance.update(dets=det)
            all_tracked_objs = self.tracker_instance.get_tracked_objects()
            
            states = [obj.get_state()[0] for obj in tracked_objs]
            ids = [obj.id for obj in tracked_objs] 
            cls_ids = [obj.class_id for obj in tracked_objs]
            
            if len(states) == 0:
                sv_detections = sv.Detections.empty()
            else:
                xyxy = np.array(states)
                tracker_ids = np.array(ids)
                tracker_cls_ids = np.array(cls_ids)
                sv_detections = sv.Detections(
                    xyxy=xyxy,
                    tracker_id=tracker_ids,
                    class_id=tracker_cls_ids
                )
            
            visualized_tracked_objs, visualized_sv_detections = self.filter_vehicles_in_zone(tracked_objs, all_tracked_objs, sv_detections, frame_counter, buffer_maxlen)

            # Update frame buffer
            frame_buffer.append((frame_counter, frame.copy()))
            
            # Detect traffic light states
            if light_detector is not None and light_fsm is not None:
                # Detect every 10 frames to improve FPS
                if frame_counter % 10 == 0:
                    detected_lights = light_detector.detect_light_signals(frame)
                    traffic_light_states = light_fsm.update(candidates=detected_lights, frame_idx=frame_counter)
                else:
                    traffic_light_states = light_fsm.get_states()
            else:
                # Fallback to hardcoded values if no light zones configured
                traffic_light_states = [None, 'RED', None]
            
            # Violation Update
            stats = self.violation_manager.update(
                vehicles=visualized_tracked_objs, 
                sv_detections=visualized_sv_detections, 
                frame=frame, 
                traffic_light_state=traffic_light_states, 
                frame_buffer=frame_buffer, 
                fps=FPS, 
                save_queue=self.violation_queue
            )
            
            # Draw
            annotated_frame = render_frame(visualized_tracked_objs, frame, visualized_sv_detections, self.box_annotator, self.label_annotator)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            yield annotated_frame, stats

    def get_latest_frame(self):
        if self.generator:
            try:
                return next(self.generator)
            except StopIteration:
                self.running = False
                return None, None
        return None, None
