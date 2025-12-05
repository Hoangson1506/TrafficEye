import cv2
import argparse

# Point to one image and its corresponding label file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeled Data Visualization")
    parser.add_argument("--img_path", type=str, default="data/YOLO/images/train/MOT16-02_000059.jpg", help="Path to image file")
    parser.add_argument("--label_path", type=str, default="data/YOLO/labels/train/MOT16-02_000059.txt", help="Path to label file, YOLO format")
    args = parser.parse_args()
    img_path = args.img_path
    label_path = args.label_path
    # Load image
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    def draw_yolo_bbox(img_path, label_file):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.strip().split())

            # Convert YOLO normalized → pixel coordinates
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            cls = int(cls)
            label = "pedestrian"

            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

    visualized = draw_yolo_bbox(img_path, label_path)
    cv2.imshow("YOLO Dataset Visualization", visualized)
    key = cv2.waitKey(0)
    if key == 10:   # ESC key code
        cv2.destroyAllWindows()