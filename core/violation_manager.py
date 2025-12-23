from typing import List
from core.violation import Violation
from core.vehicle import Vehicle
from core.license_plate_recognizer import LicensePlateRecognizer
from supervision import Detections

class ViolationManager:
    """
    Manage violation of tracked vehicles
    """
    def __init__(self, violations: List[Violation], recognizer: LicensePlateRecognizer, lp_detection_interval: int = 5, **kwargs):
        self.violation_count = {violation.name: 0 for violation in violations}
        self.violations = violations
        self.recognizer = recognizer
        self.frame_counter = 0
        self.lp_detection_interval = lp_detection_interval

    def update(self, vehicles: List[Vehicle], sv_detections: Detections, frame, traffic_light_state, **kwargs):
        """
        Update violation of tracked vehicles

        Args:
            vehicles (List[Vehicle]): List of tracked vehicles
        """
        self.frame_counter += 1

        # Centralized continuous license plate detection for ALL violated vehicles
        # Only run every N frames to improve performance
        if self.frame_counter % self.lp_detection_interval == 0:
            for vehicle in vehicles:
                if vehicle.has_violated is True:
                    current_state = vehicle.get_state()[0]
                    candidate_lp = self.recognizer.update(frame, current_state)
                    vehicle.update_license_plate(candidate_lp)

        # Check all violation types
        for violation in self.violations:
            self.violation_count[violation.name] += len(violation.check_violation(vehicles, sv_detections, frame, traffic_light_state, **kwargs))

        return self.violation_count