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

        self.lp_votes = {}
        self.license_plate = None
        self.vote_threshold = 3

        self.proof = []


    def update_license_plate(self, candidate):
        if candidate is None:
            return None
        
        self.lp_votes[candidate] = self.lp_votes.get(candidate, 0) + 1

        best_plate = max(self.lp_votes, key=self.lp_votes.get)

        if self.lp_votes[best_plate] >= self.vote_threshold:
            self.license_plate = best_plate

        return self.license_plate
    

    def mark_violation(self, violation_type, recognizer, frame=None, padding=None,
                       frame_buffer=None, fps=30, save_queue=None):

        if padding is None:
            padding = config['violation']['padding']

        candidate_lp = recognizer.update(self, frame)

        final_lp = self.update_license_plate(candidate_lp)

        if final_lp is None:
            return  # Wait until 3 votes

        if not self.has_violated:
            self.has_violated = True
            self.violation_type.append(violation_type)
            self.violation_time.append(time.time())

            if frame is not None:
                x1, y1, x2, y2 = map(int, self.get_state()[0])
                h, w, _ = frame.shape

                self.proof = frame[max(0, y1 - padding):min(h, y2 + padding),
                                   max(0, x1 - padding):min(w, x2 + padding)].copy()

                violation_data = {
                    'vehicle_id': self.id,
                    'identifier': final_lp,
                    'violation_type': violation_type,
                    'frame': frame.copy(),
                    'bbox': (x1, y1, x2, y2),
                    'frame_buffer': list(frame_buffer) if frame_buffer else [],
                    'fps': fps,
                    'proof_crop': self.proof
                }
                if save_queue is not None:
                    save_queue.put(violation_data)
