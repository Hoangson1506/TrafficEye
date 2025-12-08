from ultralytics import YOLO
from track.sort import SORT
from track.bytetrack import ByteTrack
from detect.detect import inference_video
from core.vehicle import Vehicle
from utils.parse_args import parse_args_tracking
from utils.drawing import draw_polygon_zone, draw_and_write_frame
from utils.io import handle_result_filename, handle_video_capture
from detect.utils import preprocess_detection_result
from core.violation import RedLightViolation
from core.violation_manager import ViolationManager
from utils.config import load_config
import cv2
import numpy as np
import supervision as sv
import os
import csv
from collections import deque


if __name__ == "__main__":
    args = parse_args_tracking()

    # Prepare output paths
    result_filename, ext = handle_result_filename(args.data_path, args.tracker)
    video_result_path = os.path.join(args.output_dir, "video", result_filename + ext)
    csv_result_path = os.path.join(args.output_dir, "csv", result_filename + ".csv")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)

    # Load config
    config = load_config()
    
    # Override config with CLI args if provided (optional, but good practice)
    # For now, we prioritize CLI args for tracker selection, but use config for params
    
    if args.tracker == 'sort':
        cfg = config['tracking']['sort']
        tracker_instance = SORT(
            cost_function=cfg['cost_function'], 
            max_age=cfg['max_age'], 
            min_hits=cfg['min_hits'], 
            iou_threshold=cfg['iou_threshold'], 
            tracker_class=Vehicle
        )
        conf_threshold = cfg['conf_threshold']
    elif args.tracker == 'bytetrack':
        cfg = config['tracking']['bytetrack']
        tracker_instance = ByteTrack(
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
    else:
        raise ValueError(f"Unknown tracker: {args.tracker}")

    data_path = args.data_path
    model = YOLO(args.model, task='detect', verbose=True)
    device = args.device
    np.random.seed(42)

    # Setup VideoWriter and display Window
    window_name = "Traffic Violation Detection"
    FRAME_WIDTH, FRAME_HEIGHT, FPS, first_frame, ret = handle_video_capture(data_path)
    polygon_points = draw_polygon_zone(first_frame, window_name)
    polygon_points = np.array(polygon_points, dtype=int)
    polygon_zone = sv.PolygonZone(polygon_points) if len(polygon_points) >= 3 else None

    # supervision annotator for visualization
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
    line_zone_annotator = sv.LineZoneAnnotator()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_result_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

    # Prepare detections
    dets = inference_video(
        model=model,
        data_path=data_path,
        output_path=None,
        device=device,
        stream=True,
        conf_threshold=conf_threshold,
        classes=config['detections']['classes'],
        imgsz=config['detections']['imgsz'],
        iou_threshold=config['detections']['iou_threshold']
    )
    csv_results = []
    
    # Frame buffer for video proof
    buffer_duration = config['violation']['video_proof_duration']
    buffer_maxlen = int(FPS * buffer_duration)
    frame_buffer = deque(maxlen=buffer_maxlen)

    # Set up violation manager and violation types
    violations = [RedLightViolation(frame=first_frame, window_name=window_name)]
    violation_manager = ViolationManager(violations=violations)

    for i, result in enumerate(dets):
        frame, det = preprocess_detection_result(result, polygon_zone)

        # Object tracking
        tracked_objs = tracker_instance.update(dets=det)

        # Can optimize for better performance, now using 2 for loops for simplicity
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
        
        # Update frame buffer
        frame_buffer.append(frame.copy())

        # Update violation manager
        violation_manager.update(vehicles=tracked_objs, sv_detections=sv_detections, frame=frame, frame_buffer=frame_buffer, fps=FPS)
        
        draw_and_write_frame(tracked_objs, frame, sv_detections, box_annotator, label_annotator, video_writer)

        if args.save == "True":
            frame_num = i + 1
            for obj in tracked_objs:
                x1, y1, x2, y2 = map(float, obj.get_state()[0])
                t_id = int(obj.id)
                violated = 1 if getattr(obj, 'has_violated', False) else 0

                csv_results.append([frame_num, x1, y1, x2, y2, t_id, violated])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    if args.save == "True":
        with open(csv_result_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_results)
    print(f"Tracking results succesfully saved to {video_result_path} and {csv_result_path}")
    print(FRAME_WIDTH, FRAME_HEIGHT, FPS)
    print(len(csv_results))