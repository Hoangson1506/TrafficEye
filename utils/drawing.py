import cv2
import numpy as np
import supervision as sv

def draw_polygon_zone(frame: np.ndarray, window_name: str = "Traffic Violation Detection"):
    """
    Interactive mode to draw a Polygon Zone (Region of Interest, RoI).
    
    The user uses mouse clicks to define vertices.
    Press 'n' to save the polygon (requires ≥ 3 points). 
    Press 'ESC' to cancel.

    Args:
        frame (np.ndarray): The video frame used as the canvas.

    Return:
        drawing_points: A list of points defining the RoI polygon (N, 2). 
                    Returns [] if cancelled.
    """
    
    zone_name = "POLYGON_ROI"
    drawing_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose POLYGON points (RoI Region)")
    print("2. Press 's' to save the polygon (Needs ≥ 3 points)")
    print("3. Press 'r' to revert back 1 point")
    print("3. Press 'ESC' to cancel")
    print("Warning: At least 3 points are needed for a polygon. If you save with less than 3 points, it will return None.")


    while True:
        display_frame = frame.copy()

        # Draw selected points
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)

        # Draw the polygon if enough points are selected (>= 3)
        if len(drawing_points) >= 3:
            pts = np.array(drawing_points, np.int32).reshape((-1, 1, 2))
            # Draw the connected, closed polygon (True)
            cv2.polylines(display_frame, [pts], True, (255, 0, 0), 2)
        
        cv2.putText(display_frame,
                    f"MODE: {zone_name} | Points: {len(drawing_points)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print(f"ESC pressed → {zone_name} cancelled")
            cv2.destroyWindow(window_name)
            return []
        
        if key == ord('r'):
            if len(drawing_points) > 0:
                removed_point = drawing_points.pop()
                print(f"[{zone_name}] Removed last point: {removed_point}")
            else:
                print(f"[{zone_name}] No points to remove!")

        if key == ord('s'):
            if len(drawing_points) < 3:
                print(f"[{zone_name}] Requires at least 3 points for a polygon!")
                cv2.destroyWindow(window_name)
                return drawing_points
            
            print(f"[{zone_name}] Polygon saved with {len(drawing_points)} points.")
            cv2.destroyWindow(window_name)
            return drawing_points


def draw_line_zone(frame: np.ndarray, zone_name: str = "LINE_ZONE", window_name: str = "Traffic Violation Detection"):
    """
    Interactive mode to draw a Line Zone (Boundary/Crossing Line).
    
    The user must define exactly 2 points per line.
    Press 's' to save the lines. Press 'ESC' to cancel.

    Args:
        frame (np.ndarray): The video frame used as the canvas.
        zone_name (str): Name of the zone type being drawn (e.g. "Violation Lines")

    Return:
        drawing_points: A list of points defining the lines. Returns [] if cancelled.
    """
    
    drawing_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_points.append((x, y))
            print(f"[{zone_name}] Point selected: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose 2 points for each LINE")
    print("2. Press 'r' to revert back 1 point")
    print("3. Press 'ESC' to cancel (Returns empty list)")
    print("4. Press 's' to save the lines and finish.")

    while True:
        display_frame = frame.copy()

        # Draw selected points
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

        # Draw the lines if pairs of points are selected
        if len(drawing_points) >= 2:
            for j in range(0, len(drawing_points)-1, 2):
                cv2.line(display_frame, drawing_points[j], drawing_points[j+1], (0, 0, 255), 2)

        cv2.putText(display_frame,
                    f"MODE: {zone_name} | Points: {len(drawing_points)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: # ESC
            print(f"ESC pressed → {zone_name} cancelled")
            cv2.destroyWindow(window_name)
            return []
        
        if key == ord('r'):
            if len(drawing_points) > 0:
                removed_point = drawing_points.pop()
                print(f"[{zone_name}] Removed last point: {removed_point}")
            else:
                print(f"[{zone_name}] No points to remove!")
        
        if key == ord('s'):
            if len(drawing_points) > 0 and len(drawing_points) % 2 != 0:
                print(f"[{zone_name}] Requires even number of points for line pairs! Add one more point or remove one.")
            else:
                print(f"[{zone_name}] Lines saved.")
                cv2.destroyWindow(window_name)
                return drawing_points


def render_frame(tracked_objs, frame, sv_detections, box_annotator, label_annotator, window_name="Traffic Violation Detection"):
    """Process a single detection result, draws bbox, writes the frame

    Args:
        tracked_objs (KalmanBoxTracker): List of tracked objects 
        frame (ArrayLike): The frame to write on
        sv_detections (sv.Detections): Detections result in the supervision format
        box_annotator (sv.BoxAnnotator)
        label_annotator (sv.LabelAnnotator)
    """
    frame = box_annotator.annotate(
        scene=frame,
        detections=sv_detections
    )

    labels = [f"ID: {obj.id} {'[VIOLATION]' if obj.has_violated else ''}" for obj in tracked_objs]
    frame = label_annotator.annotate(
        scene=frame,
        detections=sv_detections,
        labels=labels
    )

    cv2.imshow(window_name, frame)
