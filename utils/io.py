import os
import re
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



def violation_save_worker(save_queue):
    client = MinioClient()

    while True:
        data = save_queue.get()
        if data is None: # Nhận tín hiệu dừng
            break
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
            client.save_labeled_proof(frame, identifier, violation_type, bbox)
            
            if frame_buffer:
                client.save_video_proof(frame_buffer, identifier, violation_type, fps)
            
            print(f"[Worker] Saved violation for ID: {identifier}")
        except Exception as e:
            print(f"[Worker] Error saving violation: {e}")
        finally:
            save_queue.task_done()