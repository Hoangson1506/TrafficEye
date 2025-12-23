from track.kalman_box_tracker import KalmanBoxTracker
import time
from utils import MinioClient, load_config

# Load config once
config = load_config()

class Vehicle(KalmanBoxTracker):
    def __init__(self, bbox, class_id=None, **kwargs):
        super().__init__(bbox, **kwargs)

        self.class_id = int(class_id)

        self.is_being_tracked = False
        self.has_violated = False
        self.going_straight = True
        self.frame_of_violation = None
        self.state_when_violation = None
        self.bboxes_buffer = []
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
    

    def mark_violation(self, violation_type, frame=None, padding=None,
                       frame_buffer=None, bboxes_buffer=None, fps=30, state=None, save_queue=None):

        if padding is None:
            padding = config['violation']['padding']            

        # Use already-accumulated license plate votes (from continuous detection)
        # If threshold was met, self.license_plate is set; otherwise get best candidate
        if self.license_plate is not None:
            final_lp = self.license_plate
        elif self.lp_votes:
            # Threshold not met, but we have candidates - use the best one
            final_lp = max(self.lp_votes, key=self.lp_votes.get)
        else:
            # No candidates at all - fall back to UNIDENTIFIED
            final_lp = "UNIDENTIFIED"

        if self.has_violated is True:
            self.has_violated = None
            self.violation_type.append(violation_type)
            self.violation_time.append(time.time())

            if frame is not None:
                x1, y1, x2, y2 = map(int, state)
                h, w, _ = frame.shape

                self.proof = frame[max(0, y1 - padding):min(h, y2 + padding),
                                   max(0, x1 - padding):min(w, x2 + padding)].copy()

                violation_data = {
                    'vehicle_id': self.id,
                    'identifier': final_lp,
                    'violation_type': violation_type,
                    'frame': frame.copy(),
                    'bbox': (x1, y1, x2, y2),
                    'bboxes': bboxes_buffer,
                    'frame_buffer': list(frame_buffer) if frame_buffer else [],
                    'fps': fps,
                    'proof_crop': self.proof
                }
                if save_queue is not None:
                    save_queue.put(violation_data)
