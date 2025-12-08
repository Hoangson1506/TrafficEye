from track.kalman_box_tracker import KalmanBoxTracker
import time
from utils.storage import MinioClient
from utils.config import load_config

# Load config once
config = load_config()

class Vehicle(KalmanBoxTracker):
    def __init__(self, bbox, class_id=None, **kwargs):
        super().__init__(bbox, **kwargs)

        self.class_id = int(class_id)
        self.has_violated = False
        self.violation_type = []
        self.violation_time = []
        self.license_plate = None
        self.proof = [] # np.array (Crop images)

    def mark_violation(self, violation_type, frame=None, padding=None, frame_buffer=None, fps=30):
        if padding is None:
            padding = config['violation']['padding']
        if not self.has_violated:
            self.has_violated = True
            self.violation_type.append(violation_type)
            self.violation_time.append(time.time())

            if frame is not None:
                x1, y1, x2, y2 = map(int, self.get_state()[0])
                h, w, _ = frame.shape

                self.proof = frame[max(0, y1 - padding):min(h, y2 + padding),
                                   max(0, x1 - padding):min(w, x2 + padding)].copy()
                
                # Save proof and retraining data
                MinioClient().save_proof(self.proof, self.id, violation_type)
                MinioClient().save_retraining_data(frame, self.id, (x1, y1, x2, y2))
                
                # Save labeled proof
                MinioClient().save_labeled_proof(frame, self.id, violation_type, (x1, y1, x2, y2))
                
                # Save video proof
                if frame_buffer:
                    MinioClient().save_video_proof(list(frame_buffer), self.id, violation_type, fps)
