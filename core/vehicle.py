from track.kalman_box_tracker import KalmanBoxTracker
import time

class Vehicle(KalmanBoxTracker):
    def __init__(self, bbox, class_id=None, **kwargs):
        super().__init__(bbox, **kwargs)

        self.class_id = int(class_id)
        self.has_violated = False
        self.violation_type = []
        self.violation_time = []
        self.license_plate = None
        self.proof = [] # np.array (Crop images)

    def mark_violation(self, violation_type, frame=None, padding=30):
        if not self.has_violated:
            self.has_violated = True
            self.violation_type.append(violation_type)
            self.violation_time.append(time.time())

            if frame is not None:
                x1, y1, x2, y2 = map(int, self.get_state()[0])
                h, w, _ = frame.shape

                self.proof = frame[max(0, y1 - padding):min(h, y2 + padding),
                                   max(0, x1 - padding):min(w, x2 + padding)].copy()
