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

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose POLYGON points (RoI Region)")
    print("2. Press 'n' to save the polygon (Needs ≥ 3 points)")
    print("3. Press 'ESC' to cancel")

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

        if key == ord('n'):
            if len(drawing_points) < 3:
                print(f"[{zone_name}] Requires at least 3 points for a polygon!")
                continue
            
            print(f"[{zone_name}] Polygon saved with {len(drawing_points)} points.")
            cv2.destroyWindow(window_name)
            return drawing_points

# --- FUNCTION 2: DRAW LINE ZONE (Boundary Line) ---

def draw_line_zone(frame: np.ndarray, window_name: str = "Traffic Violation Detection"):
    """
    Interactive mode to draw a Line Zone (Boundary/Crossing Line).
    
    The user must define exactly 2 points.
    Press 'q' to save the line. Press 'ESC' to cancel.

    Args:
        frame (np.ndarray): The video frame used as the canvas.

    Return:
        drawing_points: A list of 2 points defining the line. Returns [] if cancelled.
    """
    
    zone_name = "LINE_ZONE"
    drawing_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points
        if event == cv2.EVENT_LBUTTONDOWN:
            # Only allow selecting up to 2 points
            if len(drawing_points) < 2:
                drawing_points.append((x, y))
                print(f"[{zone_name}] Point selected: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n--- [TUTORIAL] {zone_name} ---")
    print("1. Left mouse click to choose 2 points for the LINE")
    print("2. Press 'q' to save the line")
    print("3. Press 'ESC' to cancel")

    while True:
        display_frame = frame.copy()

        # Draw selected points
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

        # Draw the line if 2 points are selected
        if len(drawing_points) == 2:
            cv2.line(display_frame, drawing_points[0], drawing_points[1], (0, 0, 255), 2)
        
        cv2.putText(display_frame,
                    f"MODE: {zone_name} | Points: {len(drawing_points)}/2",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: # ESC
            print(f"ESC pressed → {zone_name} cancelled")
            cv2.destroyWindow(window_name)
            return []

        if key == ord('q'):
            if len(drawing_points) != 2:
                print(f"[{zone_name}] Requires exactly 2 points for a line!")
                continue
            
            print(f"[{zone_name}] Line saved.")
            cv2.destroyWindow(window_name)
            return drawing_points


def draw_and_write_frame(tracked_objs, frame, sv_detections, box_annotator, label_annotator, video_writer, window_name="Traffic Violation Detection"):
    """Process a single detection result, draws bbox, writes the frame

    Args:
        tracked_objs (KalmanBoxTracker): List of tracked objects 
        frame (ArrayLike): The frame to write on
        sv_detections (sv.Detections): Detections result in the supervision format
        box_annotator (sv.BoxAnnotator)
        label_annotator (sv.LabelAnnotator)
        video_writer (cv2.VideoWriter)
    """
    # violation_palette = sv.ColorPalette(colors=[sv.Color.GREEN, sv.Color.RED])
    # class_ids = [1 if obj.has_violated else 0 for obj in tracked_objs]
    # if len(tracked_objs) > 0:
    #     class_ids = np.array([1 if obj.has_violated else 0 for obj in tracked_objs], dtype=int)
    #     sv_detections.class_id = class_ids
    # else:
    #     sv_detections.class_id = np.array([], dtype=int)

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

    # line_zone_annotator.annotate(frame, line_counter=line_zone)

    if video_writer is not None:
        video_writer.write(frame)

    cv2.imshow(window_name, frame)

    # if len(tracked_objs) > 0:
    #     for tracker in tracked_objs:
    #         x1, y1, x2, y2 = tracker.get_state()[0]
    #         track_id = int(tracker.id)
    #         is_violated = getattr(tracker, 'has_violated', False)
    #         if is_violated:
    #             color = (0, 0, 255) # Đỏ
    #             label = f"ID: {track_id} [VIOLATION]"
    #         else:
    #             color = ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)
    #             label = f"ID: {track_id}"
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # if line_points is not None:
    #     cv2.line(frame, line_points[0], line_points[1], color=(126, 0, 126), thickness=3)
    # video_writer.write(frame)
    # cv2.imshow("Tracking Results", frame)
