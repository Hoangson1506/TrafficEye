import supervision as sv
import numpy as np
from core.vehicle import Vehicle
from typing import List
from utils.drawing import draw_line_zone

class Violation:
    """Base class of all type of traffic violations
    """
    def __init__(self, name: str, polygon_points: list, **kwargs):
        self.name = name
        self.polygon_zone = sv.PolygonZone(polygon=polygon_points, triggering_anchors=[sv.Position.CENTER])

    def check_violation(self, recognizer, vehicles: List[Vehicle]):
        """Check violation of vehicles

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
        """
        raise NotImplementedError


class RedLightViolation(Violation):
    """Red light violation"""
    def __init__(self, polygon_points, **kwargs):
        super().__init__(name="RedLightViolation", polygon_points=polygon_points)
        self.violation_lines = [] # vehicles crossing these lines will be marked as violated no matter what the traffic light state is
        # used for special cases like no U-turn allowed no matter what the traffic light state is
        # vehicles crossing these lines will be marked as violated no matter what the traffic light state is
        self.special_violation_lines = [] 

        # violated vehicles crossing exception lines will be marked as not violated
        # Vehicles are only exempted from violation if they cross these lines when the corresponding traffic light is NOT red
        # If your case requires no left turn whatever the traffic light state is, you can use this with left_light = 'RED' always
        # But better just use special_violation_lines for that case
        self.left_exception_lines = [] # for checking left turn exception
        self.right_exception_lines = [] # for checking right turn exception
        self.other_exception_lines = [] # for checking other exceptions (e.g., U-turn)
        self.draw_line(kwargs.get('frame', None), kwargs.get('window_name', "Traffic Violation Detection"))

    def check_violation(self, recognizer, vehicles: List[Vehicle], sv_detections: sv.Detections, frame, traffic_light_state: list=[None, 'RED', 'GREEN'], **kwargs):
        """Check the violation state of vehicles tracked

        Args:
            vehicles (List[Vehicle]): List of vehicles to check
            sv_detections (sv.Detections): The detection results in supervision format
            traffic_light_state (list): State of the traffic 3 lights (if exist): turn left, go straight and turn right ("RED", "GREEN", "YELLOW")
        """
        # This assumes Vietnam traffic light system with 3 lights: left turn, straight, right turn and turning laws.
        # If turning right is always allowed, always set right_Light to 'GREEN'. 
        # If no light state for turning right is provided, right_Light will be set to 'GREEN'.
        # If a vehicle can only turn right when the straight light is green because there is no right turn light, set right_Light = straight_Light.
        # If no light state for turning left is provided, left_Light will be set to the state of straight_Light.
        # This code now does not handle different vehicle types. So roads that allow some vehicle types to turn when the light is red are not supported.
        left_Light, straight_Light, right_Light = traffic_light_state
        if left_Light is None:
            left_Light = straight_Light
        if right_Light is None:
            right_Light = 'GREEN'
        
        n = len(vehicles)

        # Masks
        violated_mask = np.zeros(n, dtype=bool)
        special_violated_mask = np.zeros(n, dtype=bool)
        left_exception_mask = np.zeros(n, dtype=bool)
        right_exception_mask = np.zeros(n, dtype=bool)
        other_exception_mask = np.zeros(n, dtype=bool)
        turning_blocked_mask = np.zeros(n, dtype=bool)

        # check violation lines crossing
        for line in self.violation_lines:
            crossed_in, _ = line.trigger(detections=sv_detections)
            violated_mask |= crossed_in

        # check special violation lines crossing
        for line in self.special_violation_lines:
            crossed_in, _ = line.trigger(detections=sv_detections)
            special_violated_mask |= crossed_in

        # left turn exception lines
        for line in self.left_exception_lines:
            crossed_in, _ = line.trigger(detections=sv_detections)
            if left_Light == 'RED':
                turning_blocked_mask |= crossed_in # left turn is blocked
            else:
                left_exception_mask |= crossed_in # allow left turn

        # right turn exception lines
        for line in self.right_exception_lines:
            crossed_in, _ = line.trigger(detections=sv_detections)
            if right_Light == 'RED':
                turning_blocked_mask |= crossed_in # right turn is blocked
            else:
                right_exception_mask |= crossed_in # allow right turn

        # other exception lines (always allowed), like U-turn. If U-turn is not allowed, don't add exception lines for U-turn
        for line in self.other_exception_lines:
            crossed_in, _ = line.trigger(detections=sv_detections)
            other_exception_mask |= crossed_in

        exception_mask = left_exception_mask | right_exception_mask | other_exception_mask

        # detect vehicles leaving the polygon zone
        outside_polygon_mask = ~self.polygon_zone.trigger(detections=sv_detections)

        save_queue = kwargs.get("save_queue")
        frame_buffer = kwargs.get("frame_buffer")
        fps = kwargs.get("fps", 30)
        
        violated_vehicles = []

        # State update
        for i, vehicle in enumerate(vehicles):
            # Set violation state to True right after crossing the violation line no matter what the traffic light state is
            if violated_mask[i]:
                vehicle.has_violated = True
                vehicle.straight_light_signal_when_crossing = straight_Light
                vehicle.frame_of_violation = frame.copy()

            if special_violated_mask[i]:
                vehicle.has_violated = True
                vehicle.going_straight = False
                vehicle.frame_of_violation = frame.copy()

            # allow exceptions
            if exception_mask[i] and vehicle.has_violated:
                vehicle.has_violated = False
                vehicle.going_straight = False
                vehicle.frame_of_violation = None
            
            # turning when blocked
            if turning_blocked_mask[i] and vehicle.has_violated:
                vehicle.going_straight = False
                vehicle.frame_of_violation = frame.copy()

            # decide violation when leaving the polygon zone
            if outside_polygon_mask[i] and vehicle.has_violated:
                # straight going vehicle running red light
                if vehicle.going_straight and vehicle.straight_light_signal_when_crossing == 'RED':
                    vehicle.mark_violation("Red Light", recognizer, frame=vehicle.frame_of_violation, frame_buffer=frame_buffer, fps=fps, save_queue=save_queue)
                    violated_vehicles.append(vehicle)
                # turning vehicle running red light
                elif not vehicle.going_straight:
                    vehicle.mark_violation("Red Light - Turning", recognizer, frame=vehicle.frame_of_violation, frame_buffer=frame_buffer, fps=fps, save_queue=save_queue)
                    violated_vehicles.append(vehicle)

        return violated_vehicles

    def draw_line(self, frame: np.ndarray, window_name="Traffic Violation Detection"):
        """Draw the violation line on the frame

        Args:
            frame (np.ndarray): Frame to draw the line on
        """
        # Define categories to draw: (attribute_name, zone_display_name)
        categories = [
            ("violation_lines", "Violation Lines"),
            ("special_violation_lines", "Special Violation Lines"),
            ("left_exception_lines", "Left Turn Exception Lines"),
            ("right_exception_lines", "Right Turn Exception Lines"),
            ("other_exception_lines", "Other Exception Lines")
        ]

        for attr_name, zone_name in categories:
            # Draw lines for the current category
            points = draw_line_zone(frame, zone_name=zone_name, window_name=window_name)
            
            # Get the target list from the instance
            target_list = getattr(self, attr_name)
            
            # Convert points to sv.LineZone and append to the list
            for i in range(0, len(points), 2):
                start = sv.Point(x=points[i][0], y=points[i][1])
                end = sv.Point(x=points[i+1][0], y=points[i+1][1])
                target_list.append(sv.LineZone(start=start, end=end, triggering_anchors=[sv.Position.CENTER]))