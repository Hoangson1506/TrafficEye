import os
import shutil
import configparser
import supervision as sv
import numpy as np

CLASS_ID = 0

def convert_sequence(seq_path, output_path, split, seq, min_vis=0, class_ids=[1]):
    img_dir = os.path.join(seq_path, "img1")
    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    seqinfo = os.path.join(seq_path, "seqinfo.ini")

    print(f"\n🔹 Đang xử lý: {seq_path}")

    if not os.path.exists(seqinfo):
        print(f"seqinfo.ini not found in {seq_path}, skipping.")
        return

    # Lấy kích thước ảnh
    config = configparser.ConfigParser()
    config.read(seqinfo)
    W = int(config["Sequence"]["imWidth"])
    H = int(config["Sequence"]["imHeight"])
    seq_len = int(config["Sequence"]["seqLength"])

    # output folder
    out_img_dir = os.path.join(output_path, "images", split)
    out_lbl_dir = os.path.join(output_path, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Nếu không có ground truth (TEST)
    if not os.path.exists(gt_path):
        print("⚠ Không có gt.txt")
        return

    # Đọc annotation từ GT
    anns = {}
    img_list = sorted([i for i in os.listdir(img_dir) if i.endswith(".jpg")])

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            
            # MOT Format: frame, id, left, top, width, height, conf, class, vis
            frame = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            cls_id = int(parts[7])
            vis = float(parts[8])

            # MOT Class 1 = Pedestrian. Usually we ignore cars/static objects for human tracking.
            if cls_id not in class_ids: 
                continue
            
            # Filter low visibility or invalid boxes
            if w <= 1 or h <= 1 or vis <= min_vis:
                continue

            anns.setdefault(frame, []).append((x, y, w, h))

    # 4. Process Every Frame (Including empty ones)
    for img_filename in img_list:
        # Extract frame number from filename "000001.jpg" -> 1
        try:
            frame_idx = int(os.path.splitext(img_filename)[0])
        except ValueError:
            continue

        src_img = os.path.join(img_dir, img_filename)
        
        # Rename image to include sequence name to prevent overwrite (e.g. MOT20-01_000001.jpg)
        new_name = f"{seq}_{img_filename}"
        dst_img = os.path.join(out_img_dir, new_name)
        dst_lbl = os.path.join(out_lbl_dir, new_name.replace(".jpg", ".txt"))

        # Copy Image
        shutil.copy(src_img, dst_img)

        # Write Label File (Create empty file if no boxes, crucial for YOLO negative mining)
        with open(dst_lbl, "w") as out:
            if frame_idx in anns:
                for (x, y, w, h) in anns[frame_idx]:
                    # Convert to YOLO (Normalized Center X, Y, W, H)
                    xc = (x + w/2) / W
                    yc = (y + h/2) / H
                    nw = w / W
                    nh = h / H

                    # Clamp to [0, 1]
                    xc = max(0, min(1, xc))
                    yc = max(0, min(1, yc))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    out.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

def generate_data_yaml(output_path, nc=1, names=None):
    if names is None:
        names=['pedestrian']

    data_yaml_path = os.path.join(output_path, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {output_path}\n\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {nc}\n\n")
        f.write(f"names: {names}\n")


def preprocess_detection_result(result, polygon_zone=None):
    """Preprocess the YOLO/Roboflow detection result for tracking algorithm

    Args:
        result (ArrayLike): The detection result
        polygon_zone (sv.PolygonZone): The region of interest to filter out detection results

    Return:
        frame (ArrayLike): The original frame
        det (ArrayLike): The preprocessed detection result (x1, y1, x2, y2, conf, cls_id)
    """
    frame = result.orig_img.copy()

    dets = sv.Detections.from_ultralytics(result)
    if polygon_zone is not None:
        mask = polygon_zone.trigger(detections=dets)
        dets = dets[mask]
    boxes = dets.xyxy
    conf = dets.confidence
    cls_id = dets.class_id

    if boxes is not None and len(boxes) > 0:
        det = np.hstack((boxes, conf.reshape(-1, 1), cls_id.reshape(-1, 1)))
    else:
        det = np.empty((0, 6))
    return frame, det
