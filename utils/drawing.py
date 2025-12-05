import cv2
import numpy as np
import supervision as sv

def select_zones(first_frame):
    """Interactive mode to draw line and RoI zones

    Args:
        first_frame(ArrayLike, np.ndarray): The first frame of the video

    Return:
        polygon_points (list): List of points of the RoI (N, 2)
        line_points (list): List of points of the line (2, 2)
    """
    drawing_points = []
    polygon_points = []
    line_points = []
    MODE = "POLYGON"

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing_points

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_points.append((x, y))
            print(f"Đã chọn điểm: ({x}, {y})")

    window_name = "Configuration: Draw POLYGON -> Press 'n' -> Draw LINE -> Press 'q'"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("--- Tutorial ---")
    print("1. Left mouse click to choose POLYGON points (RoI Region)")
    print("2. Press 'n' to save POLYGON and change to drawing LINE")
    print("3. Press 'q' to complete")

    while True:
        display_frame = first_frame.copy()

        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)

        if len(polygon_points) >= 3:
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (255, 0, 0), 2)

        if len(line_points) == 2:
            cv2.line(display_frame, line_points[0], line_points[1], (0, 0, 255), 2)

        cv2.putText(display_frame,
                    f"MODE: {'POLYGON' if MODE == "POLYGON" else 'LINE'}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("ESC pressed → exit")
            break

        if key == ord('n') and MODE == "POLYGON":
            if len(drawing_points) < 3:
                print("Polygon needs ≥ 3 points!")
                continue
            polygon_points = drawing_points.copy()
            drawing_points.clear()
            MODE = "LINE"
            print("Polygon saved, now to Line")

        elif key == ord('q'):
            if len(drawing_points) < 2:
                print("Line needs exactly 2 points!")
                continue
            line_points = drawing_points[:2]
            print("Line saved → Done")

    cv2.destroyAllWindows()
    return polygon_points, line_points


def draw_and_write_frame(tracked_objs, frame, sv_detections, line_zone, box_annotator, label_annotator, line_zone_annotator, video_writer):
    """Process a single detection result, draws bbox, writes the frame

    Args:
        tracked_objs (KalmanBoxTracker): List of tracked objects 
        frame (ArrayLike): The frame to write on
        sv_detections (sv.Detections): Detections result in the supervision format
        line_zone (sv.LineZone): supervision line zone
        box_annotator (sv.BoxAnnotator)
        label_annotator (sv.LabelAnnotator)
        line_zone_annotator (sv.LineZoneAnnotator)
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

    line_zone_annotator.annotate(frame, line_counter=line_zone)

    if video_writer is not None:
        video_writer.write(frame)

    cv2.imshow("Tracking Results", frame)

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
