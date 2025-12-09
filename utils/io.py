import os
import cv2
import re
import glob
from utils.storage import MinioClient

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


def violation_save_worker(save_queue):
    client = MinioClient()

    while True:
        data = save_queue.get()
        if data is None: # The None data is just a way to stop the worker
            break

        try:
            vehicle_id = data['vehicle_id']
            identifier = data['identifier'] # license plate or vehicle id
            violation_type = data['violation_type']
            frame = data['frame']
            bbox = data['bbox']
            frame_buffer = data['frame_buffer']
            fps = data['fps']
            proof_crop = data['proof_crop']

            client.save_proof(proof_crop, identifier, violation_type)
            client.save_retraining_data(frame, vehicle_id, bbox)
            client.save_retraining_data(frame, identifier, violation_type, bbox)
            
            if frame_buffer:
                client.save_video_proof(frame_buffer, identifier, violation_type, fps)
            
            print(f"[Worker] Saved violation for ID: {identifier}")
        except Exception as e:
            print(f"[Worker] Error saving violation: {e}")
        finally:
            save_queue.task_done()