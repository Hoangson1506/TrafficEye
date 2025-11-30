from ultralytics import YOLO
from track.sort import SORT
from track.bytetrack import ByteTrack
from detect.detect import inference_video
from track.utils import ciou 
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


if __name__ == "__main__":
    data_path = "/home/hoangsonbandon/code/Object-Tracking/data/MOT16 test video/MOT16-01-raw.mp4"
    output_path = "output"
    result_path = None
    model = YOLO("yolo12n.pt", task='detect', verbose=True)
    device = "cuda"
    np.random.seed(42)

    # Setup VideoWriter and display Window
    cap = cv2.VideoCapture(data_path)
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(result_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.namedWindow("Tracking Results", cv2.WINDOW_AUTOSIZE)

    # Prepare detections
    dets = inference_video(
        model=model,
        data_path=data_path,
        output_path=output_path,
        device=device,
        stream=True,
        classes=[0]
    )
    tracker = SORT(cost_function=ciou, max_age=30, min_hits=5, iou_threshold=0.3)
    final_results = []

    for i, result in enumerate(dets):
        frame, tracked_objs = process_and_write_frame(i, result, tracker, video_writer)
        final_results.extend(tracked_objs)
        cv2.imshow("Tracking Results", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Tracking results succesfully saved to {result_path}")
    print(FRAME_WIDTH, FRAME_HEIGHT, FPS)
    print(len(final_results))