# ---------------------------------------
# Egyptian licence plate reading (LPR / ANPR)
# ---------------------------------------
# Two small YOLO models trained on the public EALPR dataset
# (Youssef et al., 2022: Egyptian car photos with plates and every Arabic character labelled):
#   models/plate_detector.pt  finds the plate on a car
#   models/plate_chars.pt     finds each Arabic letter and digit on the plate
# Egyptian plates have the letters on the right and the number on the left:
#   letters are read right-to-left, digits left-to-right  ->  "س ط ن ٧١٢٦"
#
# For the real product at gates, Hikvision ANPR cameras read the plate inside the
# camera; Ma'at would take the plate text from the camera instead of these models.
# ---------------------------------------
import json
import os

import cv2

from core.models import INFERENCE_LOCK, _load, path

PLATE_MODEL = path("models", "plate_detector.pt")
CHAR_MODEL = path("models", "plate_chars.pt")
PLATE_CONF = 0.35
CHAR_CONF = 0.25
LETTERS = "أبجدرسصطعفقلمنھوىك"     # ك is ready for the retrained model (it is not in EALPR)
DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN = {"أ": "A", "ب": "B", "ج": "G", "د": "D", "ر": "R", "س": "S", "ص": "SS", "ط": "T", "ع": "E",
         "ف": "F", "ق": "K", "ل": "L", "م": "M", "ن": "N", "ھ": "H", "و": "W", "ى": "Y", "ك": "Q"}


def available():
    return os.path.exists(PLATE_MODEL) and os.path.exists(CHAR_MODEL)


def _plate_model():
    from ultralytics import YOLO
    return _load("plate", lambda: YOLO(PLATE_MODEL))


def _char_model():
    from ultralytics import YOLO
    return _load("plate_chars", lambda: YOLO(CHAR_MODEL))


def find_plates(image):
    """[(box, conf)] in image coordinates."""
    with INFERENCE_LOCK:
        r = _plate_model().predict(image, conf=PLATE_CONF, imgsz=352, verbose=False)[0]
    return [([float(v) for v in b.xyxy[0]], float(b.conf)) for b in r.boxes]


def read_characters(plate_img):
    """Returns (letters, digits, confidence) from a plate crop."""
    h, w = plate_img.shape[:2]
    if w < 20 or h < 8:
        return "", "", 0.0
    if w < 224:
        plate_img = cv2.resize(plate_img, None, fx=224 / w, fy=224 / w, interpolation=cv2.INTER_CUBIC)
    model = _char_model()
    with INFERENCE_LOCK:
        r = model.predict(plate_img, conf=CHAR_CONF, imgsz=224, verbose=False)[0]
    chars = []
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        chars.append((model.names[int(b.cls)], (x1 + x2) / 2, (y1 + y2) / 2, float(b.conf), x2 - x1))
    if not chars:
        return "", "", 0.0
    # drop duplicates (two classes on the same character): keep the most confident
    chars.sort(key=lambda c: -c[3])
    kept = []
    for c in chars:
        if all(abs(c[1] - k[1]) > 0.3 * min(c[4], k[4]) for k in kept):
            kept.append(c)
    # the characters sit on one line in the lower part of the plate; ignore stray marks far from it
    mid = sorted(k[2] for k in kept)[len(kept) // 2]
    kept = [k for k in kept if abs(k[2] - mid) < 0.40 * plate_img.shape[0]]
    letters = sorted((k for k in kept if k[0] not in DIGITS), key=lambda k: -k[1])   # right to left
    digits = sorted((k for k in kept if k[0] in DIGITS), key=lambda k: k[1])       # left to right
    conf = min((k[3] for k in kept), default=0.0)
    return "".join(k[0] for k in letters), "".join(k[0] for k in digits), conf


TRAINING_DIR = path("training", "plates_new")


def save_for_training(crop, plate_text):
    """Keep every plate the cameras read, with the model's own labels (YOLO format), so new letters
    such as ك can be added: fix the labels of plates that have them, then run train_plate_chars.bat."""
    try:
        h, w = crop.shape[:2]
        if w < 60:
            return None
        model = _char_model()
        with INFERENCE_LOCK:
            r = model.predict(crop, conf=CHAR_CONF, imgsz=224, verbose=False)[0]
        os.makedirs(os.path.join(TRAINING_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(TRAINING_DIR, "labels"), exist_ok=True)
        import time as _t
        stem = f"{_t.strftime('%Y%m%d_%H%M%S')}_{key(plate_text)}"
        cv2.imwrite(os.path.join(TRAINING_DIR, "images", stem + ".png"), crop)
        with open(os.path.join(TRAINING_DIR, "labels", stem + ".txt"), "w") as f:
            for b in r.boxes:
                x, y, bw, bh = [float(v) for v in b.xywhn[0]]
                f.write(f"{int(b.cls)} {x:.5f} {y:.5f} {bw:.5f} {bh:.5f}\n")
        return stem
    except Exception:           # collecting data must never break the camera
        return None


def valid(letters, digits):
    """Egyptian plates: 1-3 letters and 1-4 digits (most often 3 + 3/4; special plates can be as short as 'د ١')."""
    return 1 <= len(letters) <= 3 and 1 <= len(digits) <= 4


def text(letters, digits):
    """'س ط ن ٧١٢٦' as written on the plate."""
    return " ".join(letters) + " " + digits


def key(plate):
    """One spelling for comparing plates: no spaces, Western digits. 'س ط ن ٧١٢٦' -> 'سطن7126'."""
    plate = "".join(ch for ch in plate if not ch.isspace())
    return plate.translate(str.maketrans(DIGITS + "0123456789", "0123456789" * 2))


def latin(letters, digits):
    """'S T N 7126' - for people and systems that cannot type Arabic."""
    return " ".join(LATIN.get(ch, "?") for ch in letters) + " " + key(digits)


def read(image, box=None):
    """Find and read the best plate inside `box` (a vehicle) or the whole image.
    Returns {'plate', 'latin', 'key', 'conf', 'box', 'crop'} or None."""
    h, w = image.shape[:2]
    ox, oy, region = 0, 0, image
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = int(0.05 * (x2 - x1))
        ox, oy = max(0, x1 - pad), max(0, y1 - pad)
        region = image[oy:min(h, y2 + pad), ox:min(w, x2 + pad)]
        if region.size == 0 or region.shape[1] < 60:
            return None
    best = None
    for (px1, py1, px2, py2), pconf in find_plates(region):
        bw, bh = px2 - px1, py2 - py1
        cx1, cy1 = int(max(0, px1 - 0.06 * bw)), int(max(0, py1 - 0.1 * bh))
        cx2, cy2 = int(min(region.shape[1], px2 + 0.06 * bw)), int(min(region.shape[0], py2 + 0.1 * bh))
        crop = region[cy1:cy2, cx1:cx2]
        letters, digits, cconf = read_characters(crop)
        if not valid(letters, digits):
            continue
        conf = pconf * cconf
        if best is None or conf > best["conf"]:
            plate = text(letters, digits)
            best = {"plate": plate, "latin": latin(letters, digits), "key": key(plate), "conf": round(conf, 3),
                    "box": [ox + px1, oy + py1, ox + px2, oy + py2], "crop": crop.copy()}
    return best


class PlateVotes:
    """A plate read on several frames: the most frequent reading wins (needs 2 agreeing reads)."""
    def __init__(self):
        self.votes, self.best = {}, {}
        self.recent = []

    def new_vehicle(self):
        """True when the last two reads agree on a plate that is not the settled one:
        another car has taken this car's place (a queue at a gate)."""
        current = self.result
        return (current is not None and len(self.recent) >= 2 and self.recent[-1] == self.recent[-2]
                and self.recent[-1] != current["key"])

    def add(self, result):
        if not result:
            return
        k = result["key"]
        self.recent = (self.recent + [k])[-3:]
        self.votes[k] = self.votes.get(k, 0) + 1 + result["conf"]
        if k not in self.best or result["conf"] > self.best[k]["conf"]:
            self.best[k] = result

    @property
    def result(self):
        if not self.votes:
            return None
        k = max(self.votes, key=self.votes.get)
        return self.best[k] if self.votes[k] >= 2 else None


# ---------------------------------------
# Car lists (green / red / black), keyed by plate
# ---------------------------------------
def lists_path():
    import face_function
    return os.path.join(face_function.FACES_DIR, "plates.json")


def load_lists():
    try:
        with open(lists_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def set_list(plate, list_name, owner=""):
    if list_name not in ("green", "red", "black"):
        raise ValueError("Choose green, red or black.")
    k = key(plate)
    if len(k) < 3:
        raise ValueError("Type the plate, for example س ط ن 7126.")
    data = load_lists()
    data[k] = {"plate": normalise(plate), "list": list_name, "owner": owner.strip()[:60]}
    os.makedirs(os.path.dirname(lists_path()), exist_ok=True)
    with open(lists_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data[k]


def remove(plate):
    data = load_lists()
    data.pop(key(plate), None)
    with open(lists_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def normalise(plate):
    """Typed plates ('سطن 7126', 'س ط ن ٧١٢٦') -> 'س ط ن ٧١٢٦'."""
    k = key(plate)
    letters = "".join(ch for ch in k if not ch.isdigit())
    digits = "".join(ch for ch in k if ch.isdigit()).translate(str.maketrans("0123456789", DIGITS))
    return text(letters, digits)


def lookup(plate_key, lists=None):
    entry = (lists if lists is not None else load_lists()).get(plate_key)
    return entry or {"list": "unknown", "owner": ""}
