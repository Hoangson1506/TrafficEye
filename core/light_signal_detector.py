import cv2
import numpy as np
from utils.drawing import draw_light_zone

class LightSignalDetector:
    def __init__(self, **kwargs):
        
        self.straight_light_zones = []  # List of polygons defining straight light zones
        self.left_light_zones = []      # List of polygons defining left turn light zones
        self.right_light_zones = []     # List of polygons defining right turn light zones

        self.draw_zones(kwargs.get('frame', None), kwargs.get('window_name', "Traffic Violation Detection"))

    def detect_light_signals(self, image):
        """Detect light signals in the defined zones.

        Args:
            image (_type_): input image/frame

        Returns:
            _type_: return 3 lists of detected light signals for left, right, and straight directions. If a list is empty, it means no zones were defined for that direction.
        """
        left_lights = []
        right_lights = []
        straight_lights = []

        left_candidate = None
        straight_candidate = None
        right_candidate = None

        category = {'straight': (self.straight_light_zones, straight_lights),
                    'left': (self.left_light_zones, left_lights) if len(self.left_light_zones) > 0 else None,
                    'right': (self.right_light_zones, right_lights) if len(self.right_light_zones) > 0 else None}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        yellow1 = cv2.inRange(hsv, (5, 30, 120), (25, 255, 255))
        yellow2 = cv2.inRange(hsv, (25, 30, 120), (55, 255, 255))

        yellow_mask = cv2.bitwise_or(yellow1, yellow2)

        green_mask = cv2.inRange(hsv, (55, 70, 120), (95, 255, 255))

        for cat_name in category:
            if category[cat_name] is None:
                if cat_name == 'straight':
                    return f"ERROR: No straight light signal zones defined."
                continue
            else:
                zones, light_list = category[cat_name]
                for polygon in zones:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)

                    red_score = cv2.countNonZero(cv2.bitwise_and(red_mask, red_mask, mask=mask))
                    yellow_score = cv2.countNonZero(cv2.bitwise_and(yellow_mask, yellow_mask, mask=mask))
                    green_score = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=mask))

                    max_score = max(red_score, yellow_score, green_score)
                    if max_score == red_score:
                        light_list.append(('RED', red_score))
                    elif max_score == yellow_score:
                        light_list.append(('YELLOW', yellow_score))
                    else:
                        light_list.append(('GREEN', green_score))

        # In case there are multiple zones for a direction, and the result differs, average them by strength and pick the strongest
        for cat_name in category:
            if category[cat_name] is None:
                continue
            else:
                _, lights = category[cat_name]
                strength_dict = {}
                count_dict = {}
                for state, strength in lights:
                    strength_dict[state] = strength_dict.get(state, 0) + strength
                    count_dict[state] = count_dict.get(state, 0) + 1
                best_state = max(strength_dict, key=strength_dict.get)
                if cat_name == 'straight':
                    straight_candidate = (best_state, strength_dict[best_state] / count_dict[best_state])
                elif cat_name == 'left':
                    left_candidate = (best_state, strength_dict[best_state] / count_dict[best_state])
                else:
                    right_candidate = (best_state, strength_dict[best_state] / count_dict[best_state])

        return [left_candidate, straight_candidate, right_candidate]
    
    def draw_zones(self, frame: np.ndarray, window_name="Traffic Violation Detection"):
        """Draw light signal zones interactively.

        Args:
            frame (np.ndarray): The video frame used as the canvas.
        """
        if frame is None:
            return
        
        categories = [
            ("Straight Light Signal Zones", self.straight_light_zones),
            ("Left Turn Light Signal Zones", self.left_light_zones),
            ("Right Turn Light Signal Zones", self.right_light_zones)
        ]

        for zone_name, zone_list in categories:
            points = draw_light_zone(frame, zone_name=zone_name, window_name=window_name)
            if len(points) >= 2:
                for i in range(0, len(points), 2):
                    top_left = points[i]
                    bottom_right = points[i + 1]
                    polygon = [
                        (top_left[0], top_left[1]),
                        (bottom_right[0], top_left[1]),
                        (bottom_right[0], bottom_right[1]),
                        (top_left[0], bottom_right[1])
                    ]
                    zone_list.append(polygon)