import cv2
import warnings

warnings.filterwarnings("ignore")

class LicensePlateRecognizer:
    """
    Recognize license plate of violated vehicles
    """
    def __init__(self, license_model, character_model):
        self.license_model = license_model
        self.character_model = character_model

    def update(self, frame, state):
        """
        Detect + OCR license plate for a single vehicle
        Returns candidate license plate string (NOT final)
        """
        print('update_license')
        if frame is None:
            return None

        x1, y1, x2, y2 = map(int, state)
        h, w, _ = frame.shape

        crop = frame[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2)
        ].copy()

        if crop.size == 0:
            return None

        results = self.license_model.predict(crop, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            print('0 results')
            return None

        box = max(results[0].boxes, key=lambda b: b.conf)[0]
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].cpu().numpy())
        lp_crop = crop[ly1:ly2, lx1:lx2]

        if lp_crop.size == 0:
            return None

        plate_text = self._ocr(lp_crop)

        if plate_text is None or len(plate_text) <= 3:
            print("Cannot recognize plate")
            return None
        print(plate_text)
        return plate_text


    def _ocr(self, lp_img, min_score=0.5):
        lp_rgb = cv2.cvtColor(lp_img, cv2.COLOR_BGR2RGB)

        results = self.character_model.predict(lp_rgb)
        if not results:
            return ""

        res = results[0]

        texts  = res["rec_texts"]
        scores = res["rec_scores"]
        boxes  = res["rec_boxes"]

        if not texts:
            return ""

        merged = []
        for text, score, box in zip(texts, scores, boxes):
            if score < min_score:
                continue

            x1, y1, x2, y2 = box
            merged.append((text, score, y1, x1))

        if not merged:
            return ""

        merged_sorted = sorted(merged, key=lambda x: (x[2], x[3]))

        sorted_texts = [m[0] for m in merged_sorted]

        final_text = "".join(sorted_texts)

        return final_text