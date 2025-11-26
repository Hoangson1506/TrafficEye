from ultralytics import YOLO
import cv2
import csv
import os

def detect_person_to_csv_with_preview(
    model_path,
    video_path,
    output_csv,
    conf_thres=0.25,
    max_preview=5  # số frame muốn xem
):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    os.makedirs("preview_frames", exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h", "confidence"])

        frame_idx = 0
        preview_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            results = model(
                frame,
                conf=conf_thres,
                classes=[0]  # chỉ person
            )

            res = results[0]
            boxes = res.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Vẽ box lên frame (copy để tránh sửa frame gốc nếu cần)
            draw_frame = frame.copy()

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                w = x2 - x1
                h = y2 - y1

                # ghi CSV
                writer.writerow([
                    frame_idx,
                    float(x1),
                    float(y1),
                    float(w),
                    float(h),
                    conf
                ])

                # vẽ box
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(draw_frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                cv2.putText(
                    draw_frame,
                    f"person {conf:.2f}",
                    (x1_i, max(0, y1_i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # chỉ preview một vài frame đầu
            if preview_count < max_preview:
                preview_count += 1
                # 2) Lưu lại frame ra file
                out_path = os.path.join("preview_frames", f"frame_{frame_idx}.jpg")
                cv2.imwrite(out_path, draw_frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"✔ Đã lưu kết quả CSV: {output_csv}")
    print(f"✔ Các frame preview nằm trong folder: preview_frames/")

path = r"C:\Users\Admin\Downloads\subway.mp4"
video_path = path
model_path = "yolo11n.pt"
output_csv = "results_person.csv"

detect_person_to_csv_with_preview(
    model_path,
    video_path,
    output_csv,
    conf_thres=0.3,
    max_preview=5
)
