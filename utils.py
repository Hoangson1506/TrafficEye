import argparse
import cv2
import numpy as np
import os
import re
import glob
import supervision as sv

def select_zones(first_frame):
    """Interactive mode to draw line and RoI zones

    Args:
        first_frame(ArrayLike, np.ndarray): The first frame of the video
    """
    drawing_points = []
    polygon_points = []
    line_points = []
    MODE = "POLYGON"

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_points.append((x, y))
            print(f"Đã chọn điểm: ({x}, {y})")

    window_name = "Configuration: Draw POLYGON -> Press 'n' -> Draw LINE -> Press 'q'"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Tutorial ---")
    print("1. Left mouse click to choose POLYGON points (RoI Region)")
    print("2. Press 'n' to save POLYGON and change to drawing LINE")
    print("3. Press 'q' to complete")

    while True:
        display_frame = first_frame.copy()

        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)

        if len(polygon_points) >= 3:
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (255, 0, 0), 2)

        if len(line_points) == 2:
            cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)

        cv2.putText(display_frame,
                    f"MODE: {'POLYGON' if MODE == "POLYGON" else 'LINE'}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("ESC pressed → exit")
            break

        if key == ord('n') and MODE == "POLYGON":
            if len(drawing_points) < 3:
                print("Polygon needs ≥ 3 points!")
                continue
            polygon_points = drawing_points.copy()
            drawing_points.clear()
            MODE = "LINE"
            print("Polygon saved, now to Line")

        elif key == ord('q'):
            if len(drawing_points) < 2:
                print("Line needs exactly 2 points!")
                continue
            line_points = drawing_points[:2]
            print("Line saved → Done")

    cv2.destroyAllWindows()
    return polygon_points, line_points

def preprocess_detection_result(result, polygon_zone):
    """Preprocess the YOLO/Roboflow detection result for tracking algorithm

    Args:
        result (ArrayLike): The detection result
    """
    frame = result.orig_img.copy()

    dets = sv.Detections.from_ultralytics(result)
    mask = polygon_zone.trigger(detections=dets)
    dets = dets[mask]
    boxes = dets.xyxy
    conf = dets.confidence

    if boxes is not None and len(boxes) > 0:
        det = np.hstack((boxes, conf.reshape(-1, 1)))
    else:
        det = np.empty((0, 5))

    return frame, det

def draw_and_write_frame(tracked_objs, frame, video_writer):
    """Process a single detection result, draws bbox, writes the frame

    Args:
        tracked_objs (ArrayLike): List of tracked objects [frame_id, x1, y1, x2, y2, track_id]
        frame (ArrayLike): The frame to write on
        video_writer (VideoWriter): A cv2 VideoWriter object
    """
    if tracked_objs.size > 0:
        for track in tracked_objs:
            _, x1, y1, x2, y2, track_id = track.astype(int)
            track_id = int(track_id)
            color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    video_writer.write(frame)
    cv2.imshow("Tracking Results", frame)

def handle_result_filename(data_path, tracker_name):
    """Generate result filename based on data_path and tracker_name

    Args:
        data_path (str): path to input data
        tracker_name (str): name of the tracking algorithm
    Returns:
        str: result filename
    """
    if os.path.isdir(data_path):
        mot_pattern = re.compile(r"(MOT\d{2}-\d{2})", re.IGNORECASE)
        parts = os.path.normpath(data_path).split(os.sep)
        base_name = "result"
        for part in reversed(parts):
            match = mot_pattern.search(part)
            if match:
                base_name = match.group(1)
                break

        result_filename = f"{base_name}_{tracker_name}"
        ext = ".mp4"
        return result_filename, ext
    
    else:
        data_path_name = os.path.splitext(os.path.basename(data_path))
        base_name = data_path_name[0]
        ext = data_path_name[1]
        result_filename = f"{base_name}_{tracker_name}"
        return result_filename, ext

def handle_video_capture(data_path):
    """Handle cv2 video capture for data_path as folder and video

    Args:
        data_path (str): path to video file or folder containing images
    """
    if os.path.isdir(data_path):
        img_files = sorted(
            glob.glob(os.path.join(data_path, "*.jpg")) +
            glob.glob(os.path.join(data_path, "*.png")) + 
            glob.glob(os.path.join(data_path, "*.jpeg"))
        )

        if len(img_files) == 0:
            raise ValueError(f"No images found in directory: {data_path}")
        file_path = img_files[0]

    else:
        file_path = data_path

    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return FRAME_WIDTH, FRAME_HEIGHT, FPS, frame, ret


def parse_args_tracking():
    parser = argparse.ArgumentParser(description="Object tracking script")
    parser.add_argument(
        '--data_path', 
        type=str, 
        default="data/MOT16 test video/MOT16-01-raw.mp4",
        help='Path to the input video file (e.g., data/video.mp4).'
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo12n.pt",
        help="Path to detection model weights."
    )
    parser.add_argument(
        '--tracker', 
        type=str, 
        default='sort', 
        choices=['sort', 'bytetrack'],
        help='The tracking algorithm to use: sort or bytetrack.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='output', 
        help='Directory to save the resulting video.'
    )
    args = parser.parse_args()
    return args

def parse_args_eval():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        '--pred_path', 
        type=str, 
        default="output/csv/MOT16-02-raw_bytetrack.csv",
        help='Path to the tracking output csv file (e.g., output/csv/Name.csv).'
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="data/train/MOT16-02/gt/gt.txt",
        help="Path to ground truth text file."
    )
    parser.add_argument(
        '--min_vis', 
        type=float, 
        default=0, 
        help='min visibility to filter ground truth with low visibilities (hard to detect).'
    )
    parser.add_argument(
        '--iou_threshold', 
        type=float, 
        default=0.5, 
        help='The IoU threshold for evaluation.'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 
                 'num_false_positives', 'num_misses', 'num_switches'],
        choices=['num_frames', 'num_matches', 'num_switches', 'num_false_positives', 
                 'num_misses', 'num_detections', 'num_objects', 'num_predictions',
                 'num_unique_objects', 'mostly_tracked', 'partially_tracked',
                 'mostly_lost', 'num_fragmentations', 'motp', 'mota', 'precision',
                 'recall', 'idfp', 'idfn', 'idtp', 'idp', 'idr', 'idf1', 'obj_frequencies',
                 'pred_frequencies', 'track_ratios', 'id_global_assignment', 'deta_alpha',
                 'assa_alpha', 'hota_alpha'],
        help='metrics needed to compute. details at https://github.com/cheind/py-motmetrics'
    )
    args = parser.parse_args()
    return args