import supervision as sv
import numpy as np
from core.vehicle import Vehicle
from typing import List
from utils.drawing import draw_line_zone

class Violation:
    """Base class of all type of traffic violations
    """
    def __init__(self, name: str, **kwargs):
        self.name = name

    def check_violation(self, vehicles: List[Vehicle]):
        """Check violation of vehicles

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
        """
        raise NotImplementedError


class RedLightViolation(Violation):
    """Red light violation"""
    def __init__(self, **kwargs):
        super().__init__(name="RedLightViolation")
        self.draw_line(kwargs.get('frame', None), kwargs.get('window_name', "Traffic Violation Detection"))

    def check_violation(self, vehicles: List[Vehicle], sv_detections: sv.Detections, frame, traffic_light_state: str = "RED"):
        """Check the violation state of vehicles tracked

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
            sv_detections (sv.Detections): The detection results in supervision format
            traffic_light_state (str): State of the traffic light ("RED", "GREEN", "YELLOW")
        """
        if traffic_light_state != "RED":
            return
        
        crossed_in, crossed_out = self.stop_line.trigger(detections=sv_detections)
        is_violated_mask = crossed_in | crossed_out
        violation_indices = np.where(is_violated_mask)[0]
        violated_vehicles = []

        for i, vehicle in enumerate(vehicles):
            if i in violation_indices and hasattr(vehicle, 'mark_violation'):
                vehicle.mark_violation("Red Light", frame=frame)
                violated_vehicles.append(vehicle)

        return violated_vehicles

    def draw_line(self, frame: np.ndarray, window_name="Traffic Violation Detection"):
        """Draw the violation line on the frame

        Args:
            frame (np.ndarray): Frame to draw the line on
        """
        line_points = draw_line_zone(frame, window_name)
        start, end = sv.Point(x=line_points[0][0], y = line_points[0][1]), sv.Point(x=line_points[1][0], y = line_points[1][1])
        self.stop_line = sv.LineZone(start=start, end=end, triggering_anchors=[sv.Position.CENTER_LEFT, sv.Position.CENTER_RIGHT])