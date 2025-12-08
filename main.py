from ultralytics import YOLO
from track.sort import SORT
from track.bytetrack import ByteTrack
from detect.detect import inference_video
from track.utils import ciou, iou
from core.vehicle import Vehicle
from utils.parse_args import parse_args_tracking
from utils.drawing import draw_polygon_zone, draw_line_zone, draw_and_write_frame
from utils.io import handle_result_filename, handle_video_capture
from detect.utils import preprocess_detection_result
from core.violation import RedLightViolation
from core.violation_manager import ViolationManager
import cv2
import numpy as np
import supervision as sv
import os
import csv


if __name__ == "__main__":
    args = parse_args_tracking()

    # Prepare output paths
    result_filename, ext = handle_result_filename(args.data_path, args.tracker)
    video_result_path = os.path.join(args.output_dir, "video", result_filename + ext)
    csv_result_path = os.path.join(args.output_dir, "csv", result_filename + ".csv")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)

    if args.tracker == 'sort':
        tracker_instance = SORT(cost_function=iou, max_age=60, min_hits=5, iou_threshold=0.5, tracker_class=Vehicle)
        conf_threshold = 0.25
    elif args.tracker == 'bytetrack':
        tracker_instance = ByteTrack(cost_function=iou, max_age=60, min_hits=5, high_conf_threshold=0.5, tracker_class=Vehicle)
        conf_threshold = 0.1
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
    polygon_zone = sv.PolygonZone(polygon_points)
    # line_points = draw_line_zone(first_frame, window_name)
    # start, end = sv.Point(x=line_points[0][0], y = line_points[0][1]), sv.Point(x=line_points[1][0], y = line_points[1][1])
    # line_zone = sv.LineZone(start=start, end=end)

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
        conf_threshold=conf_threshold
    )
    csv_results = []

    # Set up violation manager and violation types
    violations = [RedLightViolation(frame=first_frame, window_name=window_name)]
    violation_manager = ViolationManager(violations=violations)

    for i, result in enumerate(dets):
        frame, det = preprocess_detection_result(result, None)

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
        
        # Update violation manager
        violation_manager.update(vehicles=tracked_objs, sv_detections=sv_detections, frame=frame)
        
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