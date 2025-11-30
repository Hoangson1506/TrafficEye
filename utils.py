import argparse
import cv2
import numpy as np

def process_and_write_frame(frame_id, result, tracker, video_writer):
    """Process a single detection result, update the tracker, draws bbox, writes the frame and returns the tracked objects

    Args:
        frame_id (int): frame index
        dets (ArrayLike): List of detections in the format [x1, y1, x2, y2, score]
        tracker (BaseTracker): A tracking algorithm instance
        video_writer (VideoWriter): A cv2 VideoWriter object
    """
    frame = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()

    if boxes is not None and len(boxes) > 0:
        det = np.hstack((boxes, conf.reshape(-1, 1)))
    else:
        det = np.empty((0, 5))

    tracked_objs = tracker.update(dets=det)
    frame_id = np.full((len(tracked_objs), 1), frame_id + 1)
    tracked_objs = np.hstack((frame_id, tracked_objs))

    if tracked_objs.size > 0:
        for track in tracked_objs:
            frame_id, x1, y1, x2, y2, track_id = track.astype(int)
            track_id = int(track_id)
            color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    video_writer.write(frame)

    return frame, tracked_objs

def parse_args():
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