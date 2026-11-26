# ---------------------------------------
# CCTV mode
# ---------------------------------------
# Runs the same AI on a recorded video file or a live camera stream
# (Hikvision / Dahua RTSP, an HTTP MJPEG stream, or a webcam number):
#   - finds people and vehicles (YOLOv8m)
#   - recognises the faces of people that were added in the glasses app
#   - follows each person across frames, so a name is decided by several
#     frames voting, not by one blurry frame
#   - logs an event the first time each person is identified, with a snapshot
#   - colours people by their list: green (allowed), red (watch), black (banned),
#     amber (unknown) - the Ma'at idea
#
# CCTV faces are small, so faces are searched inside each person box, enlarged
# 2x when needed. That finds faces about half the size a full-frame search finds.
#
# Example RTSP addresses:
#   Hikvision: rtsp://user:password@192.168.1.64:554/Streaming/Channels/101
#   Dahua:     rtsp://user:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
# ---------------------------------------
import base64
import json
import os
import threading
import time
import uuid

import cv2
import numpy as np

import activity
import lpr
import face_function as faces
from core import face_attributes, i18n
from core.models import currency_model
from core.models import INFERENCE_LOCK, object_model, path

LIST_NAMES = ("green", "red", "black")
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}

ANALYSIS_FPS = 4            # frames analysed per second of video
MAX_SIDE = 1280             # frames are shrunk to this before the AI sees them
PERSON_CONF = 0.40
VEHICLE_CONF = 0.45
TRACK_IOU = 0.25            # how much a box must overlap to be "the same person"
TRACK_TTL = 3.0             # seconds a lost person is remembered
VOTES_TO_NAME = 2           # sure face matches needed before a name is announced
FRAMES_TO_UNKNOWN = 6       # analysed frames with a face but no match -> "unknown"
MIN_SAMPLE_FACE = 24        # px in the frame; smaller faces are not saved as photos
SAMPLES_PER_PERSON = 4

COLOURS = {                 # BGR
    "green": (96, 200, 70), "red": (60, 60, 235), "black": (40, 40, 40),
    "unknown": (0, 194, 242), "vehicle": (238, 238, 238), "pending": (170, 170, 170),
}


# ---------------------------------------
# Lists (green / red / black)
# ---------------------------------------
def lists_path():
    return os.path.join(faces.FACES_DIR, "lists.json")


def load_lists():
    try:
        with open(lists_path(), encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if v in LIST_NAMES}
    except (OSError, ValueError):
        return {}


def set_list(name, list_name):
    if list_name not in LIST_NAMES:
        raise ValueError("list must be green, red or black")
    data = load_lists()
    data[name] = list_name
    os.makedirs(faces.FACES_DIR, exist_ok=True)
    with open(lists_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def list_of(name, lists):
    if not name:
        return "unknown"
    return lists.get(name, "green")    # someone added on purpose is allowed by default


# ---------------------------------------
# Small helpers
# ---------------------------------------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def shrink(frame):
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60:02d}:{seconds % 60:02d}"


def thumbnail(image, box, size=220):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    scale = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


# ---------------------------------------
# Detection
# ---------------------------------------
DEVICE_CONF = 0.30
PHONE_CONF = 0.25           # phones are small; a lower bar, checked against the hand area


def detect_objects(image):
    """People and vehicles to follow: [(label, box, conf)],
    and desk devices / phones for the working-time tracker: [(label, box)]."""
    model = object_model()
    with INFERENCE_LOCK:
        results = model.predict(image, conf=PHONE_CONF, verbose=False,
                                classes=[0, 1, 2, 3, 5, 7, 62, 63, 64, 66, 67])
    found, devices = [], []
    for r in results:
        for b in r.boxes:
            label = model.names[int(b.cls)]
            conf = float(b.conf)
            box = [float(v) for v in b.xyxy[0]]
            if label == "person":
                if conf >= PERSON_CONF:
                    found.append((label, box, conf))
            elif label in VEHICLES:
                if conf >= VEHICLE_CONF:
                    found.append((label, box, conf))
            elif label == activity.PHONE or conf >= DEVICE_CONF:
                devices.append((label, box))
    return found, devices


MOOD_MIN_FACE = 24          # px in the frame (before enlarging)
AGE_MIN_FACE = 40


def face_in_person(image, box, names, embeddings, identify=True):
    """Search the top part of a person box for a face (enlarged if small).
    Returns (face_box_in_frame or None, name, score, certainty, attributes)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # head is in the top ~45% of a standing person; add a margin for leaning
    cx1, cy1 = max(0, int(x1 - 0.1 * bw)), max(0, int(y1 - 0.05 * bh))
    cx2, cy2 = min(w, int(x2 + 0.1 * bw)), min(h, int(y1 + 0.55 * bh))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None, None, 0.0, "none", {}
    scale = 1.0
    if crop.shape[1] < 320:
        scale = min(3.0, 320 / crop.shape[1])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    found = faces.detect_faces(crop)
    if len(found) == 0:
        return None, None, 0.0, "none", {}
    face = max(found, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = [float(v) for v in face[:4]]
    frame_box = [cx1 + fx / scale, cy1 + fy / scale, cx1 + (fx + fw) / scale, cy1 + (fy + fh) / scale]
    attrs = {}
    real_width = fw / scale
    if real_width >= MOOD_MIN_FACE:
        attrs = face_attributes.analyse(crop, face)
        if real_width < AGE_MIN_FACE:
            attrs.pop("age_scores", None)
            attrs.pop("age_group", None)
    if not identify:
        return frame_box, None, 0.0, "skip", attrs
    if fw < faces.MIN_FACE_TO_RECOGNISE or len(names) == 0:
        return frame_box, None, 0.0, ("far" if len(names) else "unknown"), attrs
    ranked = faces.rank_people(faces.get_embedding(crop, face), names, embeddings)
    name, score, certainty = faces.decide(ranked)
    return frame_box, name, score, certainty, attrs


# ---------------------------------------
# Money
# ---------------------------------------
MONEY_CONF = 0.55           # higher than the glasses: a CCTV scene has more things that look like notes
MONEY_EVERY = 2             # run the currency model on every 2nd analysed frame
MONEY_STEADY = 3            # same total this many times in a row before it is logged


def detect_money(image, faces_boxes=()):
    model = currency_model()
    with INFERENCE_LOCK:
        results = model.predict(image, conf=MONEY_CONF, verbose=False)
    notes = []
    for r in results:
        for b in r.boxes:
            label = model.names[int(b.cls)]
            box = [float(v) for v in b.xyxy[0]]
            # a "note" on someone's face is a false alarm
            if any(activity.overlap(box, f) > 0.5 for f in faces_boxes):
                continue
            notes.append((label, box, float(b.conf)))
    return notes


def money_total(notes):
    total = sum(i18n.CURRENCY_VALUE.get(label, 0) for label, _, _ in notes)
    return int(total) if float(total).is_integer() else round(total, 2)


def money_breakdown(notes):
    counts = {}
    for label, _, _ in notes:
        name = i18n.CURRENCY.get(label, (label, label))[0]
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{n} × {name}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------
# Vehicle colour
# ---------------------------------------
PLATE_EVERY = 2            # read plates on every 2nd analysed frame
VEHICLE_FRAMES = 3          # a vehicle must be seen this many times before it is logged


def vehicle_colour(image, box):
    """Main paint colour of a vehicle. Windows, tyres and shadows are the darkest parts,
    so the darkest 35% of pixels are ignored before deciding."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    crop = image[max(0, y1 + h // 6):y2 - h // 6, max(0, x1 + w // 6):x2 - w // 6]
    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    v = hsv[:, 2]
    body = hsv[v >= np.percentile(v, 35)]
    hue, sat, val = body[:, 0] * 2, body[:, 1] / 255, body[:, 2] / 255
    colourful = sat > 0.35
    if colourful.mean() > 0.30:
        hh = hue[colourful]
        bins = {"red": ((hh < 15) | (hh >= 340)).sum(), "orange": ((hh >= 15) & (hh < 40)).sum(),
                "yellow": ((hh >= 40) & (hh < 70)).sum(), "green": ((hh >= 70) & (hh < 170)).sum(),
                "blue": ((hh >= 170) & (hh < 260)).sum(), "purple": ((hh >= 260) & (hh < 340)).sum()}
        name = max(bins, key=bins.get)
        if name in ("orange", "red") and np.median(val[colourful]) < 0.45:
            name = "brown" if name == "orange" else "red"
        return name
    m, tint = float(np.median(val)), float(np.median(sat))
    if m > 0.78 and tint < 0.10:
        return "white"
    return "silver" if m > 0.55 else "grey" if m > 0.32 else "black"   # a slight tint = metallic paint


# ---------------------------------------
# Tracking: who is who across frames
# ---------------------------------------
class Track:
    def __init__(self, tid, label, box, t):
        self.id, self.label, self.box = tid, label, box
        self.first, self.last = t, t
        self.votes = {}              # name -> summed score of sure matches
        self.maybe = {}
        self.unmatched = 0           # frames with a clear face that matched nobody
        self.name = None
        self.status = "pending"      # pending -> named / unknown
        self.face_box = None
        self.announced = None        # last status written to the log
        self.activity = "present"
        self.attrs = face_attributes.Votes()     # age group + mood, steadied over frames
        self.colours = {}                        # vehicle colour votes
        self.types = {label: 1}                  # vehicle type votes
        self.plates = lpr.PlateVotes()
        self.plate_box = None
        self.seen = 0
        self.snapshot = None
        self.samples = []            # [(quality, face crop)] best few, for "add this person"

    @property
    def colour(self):
        return max(self.colours, key=self.colours.get) if self.colours else None

    def update_identity(self, name, score, certainty):
        if certainty == "sure":
            self.votes[name] = self.votes.get(name, 0) + 1
            self.unmatched = max(0, self.unmatched - 1)
        elif certainty == "maybe":
            self.maybe[name] = self.maybe.get(name, 0) + 1
        elif certainty == "unknown":
            self.unmatched += 1
        if self.votes:
            best = max(self.votes, key=self.votes.get)
            if self.votes[best] >= VOTES_TO_NAME:
                self.name, self.status = best, "named"
                return
        if self.status == "pending" and self.unmatched >= FRAMES_TO_UNKNOWN:
            self.status = "unknown"


class Tracker:
    def __init__(self):
        self.tracks, self.next_id = [], 1

    def step(self, detections, t):
        unused = list(self.tracks)
        current = []
        for label, box, conf in sorted(detections, key=lambda d: -d[2]):
            best, best_score = None, 0.0
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            bw = max(1.0, box[2] - box[0])
            for tr in unused:
                if tr.label != label and not (tr.label in VEHICLES and label in VEHICLES):
                    continue
                overlap = iou(tr.box, box)
                if overlap >= TRACK_IOU:
                    score = 1 + overlap
                else:
                    # fast movers at a few frames a second may not overlap: fall back to "close by and
                    # similar size", allowing a bigger jump the longer the person was lost
                    tx, ty = (tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2
                    tw = max(1.0, tr.box[2] - tr.box[0])
                    dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / max(bw, tw)
                    size = min(bw, tw) / max(bw, tw)
                    reach = 1.2 + 1.5 * min(1.0, t - tr.last)
                    if dist > reach or size < 0.5:
                        continue
                    score = 1 - dist / (reach + 1)
                if score > best_score:
                    best, best_score = tr, score
            if best is None:
                best = Track(self.next_id, label, box, t)
                self.next_id += 1
                self.tracks.append(best)
            else:
                unused.remove(best)
                best.box, best.last = box, t
                if label in VEHICLES:               # seen from above a car can look like a bus: vote
                    best.types[label] = best.types.get(label, 0) + 1
                    best.label = max(best.types, key=best.types.get)
            current.append(best)
        self.tracks = [tr for tr in self.tracks if t - tr.last <= TRACK_TTL]
        return current


# ---------------------------------------
# A running job (one video or one stream)
# ---------------------------------------
class CCTVJob:
    def __init__(self, source, title, is_file, count_money=False):
        self.id = uuid.uuid4().hex[:10]
        self.source, self.title, self.is_file = source, title, is_file
        self.events, self.error = [], None
        self.samples = {}            # track id -> best face crops (kept after the person leaves)
        self.tracks = {}             # track id -> Track (for renaming people still on screen)
        self.activity = activity.ActivityLog()
        self.wall_start = time.time()
        self.count_money = count_money
        self.money = {"now": 0, "max": 0, "notes": "", "events": []}
        self._money_recent = []
        self.people_now, self.people_peak = 0, 0
        self.person_attrs = {}       # track id -> (age group, mood, sentiment) for the demographics
        self.state = "starting"
        self.progress = 0.0
        self.frames_analysed = 0
        self.people_seen, self.vehicles_seen = 0, 0
        self.vehicle_log = []
        self.read_plates = lpr.available()
        self.collect_plates = True       # keep plate photos for retraining (new letters like ك)
        self.latest_jpeg = None
        self.output_path = None
        self._stop = threading.Event()
        self._new_frame = threading.Condition()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---- public ----
    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return {"id": self.id, "title": self.title, "state": self.state, "error": self.error,
                    "progress": round(self.progress, 3), "frames": self.frames_analysed,
                    "people": self.people_seen, "vehicles": self.vehicles_seen,
                    "events": list(self.events),
                    "vehicle_log": [dict(v) for v in self.vehicle_log],
                    "plates_enabled": self.read_plates,
                    "activity": self.activity.summary(self.clock),
                    "now": self.people_now, "peak": self.people_peak,
                    "demographics": self.demographics(),
                    "money": dict(self.money, enabled=self.count_money),
                    "download": f"/cctv/video/{self.id}" if self.output_path and self.state == "done" else None}

    def frames(self):
        """MJPEG generator for the browser."""
        last = None
        while True:
            with self._new_frame:
                self._new_frame.wait_for(lambda: self.latest_jpeg is not last or self.state in ("done", "error", "stopped"),
                                         timeout=5)
                jpeg = self.latest_jpeg
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            elif self.state in ("done", "error", "stopped"):
                return

    # ---- worker ----
    def _publish(self, image):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._new_frame:
                self.latest_jpeg = buf.tobytes()
                self._new_frame.notify_all()

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000])
        else:
            cap = cv2.VideoCapture(src)
        return cap

    def _run(self):
        try:
            self._loop()
            with self._lock:
                if self.state == "running":
                    self.state = "stopped" if self._stop.is_set() else "done"
        except Exception as exc:          # never kill the server
            with self._lock:
                self.state, self.error = "error", str(exc)
        with self._new_frame:
            self._new_frame.notify_all()

    def _loop(self):
        cap = self._open()
        if not cap.isOpened():
            raise RuntimeError("Could not open the video. Check the file, or the camera address, user and password.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if fps <= 1 or fps > 120:
            fps = 25
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.is_file else 0
        stride = max(1, round(fps / ANALYSIS_FPS))
        names, embeddings = faces.load_face_db()
        lists = load_lists()
        tracker = Tracker()
        writer = None
        self.state = "running"

        live = LiveReader(cap) if not self.is_file else None
        index = 0
        started = time.time()
        try:
            while not self._stop.is_set():
                if live:
                    frame = live.latest()
                    if frame is None:
                        if live.dead:
                            raise RuntimeError("The camera stream stopped.")
                        time.sleep(0.05)
                        continue
                    t = time.time() - started
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index % stride:
                        index += 1
                        continue
                    t = index / fps
                    index += 1
                    if total:
                        self.progress = min(1.0, index / total)

                image = shrink(frame)
                self._full, self._scale = frame, frame.shape[1] / image.shape[1]   # plates are read at full size
                annotated = self._analyse(image, t, tracker, names, embeddings, lists)
                self._publish(annotated)
                self.frames_analysed += 1

                if self.is_file:
                    if writer is None:
                        os.makedirs(path("uploads", "cctv"), exist_ok=True)
                        self.output_path = path("uploads", "cctv", f"{self.id}.mp4")
                        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 ANALYSIS_FPS, (annotated.shape[1], annotated.shape[0]))
                    writer.write(annotated)
                    # show the video at about real speed when the computer is fast enough
                    ahead = t - (time.time() - started)
                    if ahead > 0:
                        time.sleep(min(ahead, 0.5))
        finally:
            if live:
                live.close()
            cap.release()
            if writer is not None:
                writer.release()
        if self.is_file:
            self.progress = 1.0

    def _analyse(self, image, t, tracker, names, embeddings, lists):
        lists = load_lists()            # list changes apply while the video runs
        names, embeddings = faces.load_face_db()   # people added during the video count at once
        detections, devices = detect_objects(image)
        current = tracker.step(detections, t)
        for tr in current:
            if tr.label != "person":
                continue
            if tr.status == "named":
                tr.face_box = None
                if self.frames_analysed % 3:    # name is settled; check mood/age every 3rd frame only
                    continue
                face_box, _, _, _, attrs = face_in_person(image, tr.box, names, embeddings, identify=False)
                tr.attrs.add(attrs)
                continue
            face_box, name, score, certainty, attrs = face_in_person(image, tr.box, names, embeddings)
            tr.face_box = face_box
            tr.attrs.add(attrs)
            if face_box is not None:
                self._keep_sample(image, tr, face_box)
            if face_box is not None and certainty != "far":
                tr.update_identity(name, score, certainty)
        for tr in current:
            tr.seen += 1
            if tr.label in VEHICLES:
                if tr.label != "bicycle":
                    c = vehicle_colour(image, tr.box)
                    if c:
                        tr.colours[c] = tr.colours.get(c, 0) + 1
                tr.plate_box = None
                if self.read_plates and tr.label not in ("bicycle",) and self.frames_analysed % PLATE_EVERY == 0 \
                        and sum(tr.plates.votes.values()) < 15:
                    full_box = [v * self._scale for v in tr.box]
                    got = lpr.read(self._full, full_box)
                    if got:
                        tr.plates.add(got)
                        tr.plate_box = [v / self._scale for v in got["box"]]
                        if tr.plates.new_vehicle():            # another car took this one's place
                            fresh = Track(tracker.next_id, tr.label, tr.box, t)
                            tracker.next_id += 1
                            for _ in range(2):
                                fresh.plates.add(got)
                            fresh.seen, fresh.plate_box = VEHICLE_FRAMES, tr.plate_box
                            fresh.colours = {}
                            tracker.tracks[tracker.tracks.index(tr)] = fresh
                            current[current.index(tr)] = fresh
                            tr = fresh
                area = (tr.box[2] - tr.box[0]) * (tr.box[3] - tr.box[1])
                if tr.snapshot is None or area > tr.snapshot[0]:      # keep the biggest view of it
                    tr.snapshot = (area, thumbnail(image, tr.box))
        people = [tr for tr in current if tr.label == "person"]
        with self._lock:
            self.people_now = sum(1 for tr in people if tr.seen >= VEHICLE_FRAMES or tr.status != "pending")
            self.people_peak = max(self.people_peak, self.people_now)
            for tr in people:
                if tr.attrs.mood or tr.attrs.age_group:
                    self.person_attrs[tr.id] = (tr.attrs.age_group, tr.attrs.mood, tr.attrs.sentiment)
        notes = []
        if self.count_money and self.frames_analysed % MONEY_EVERY == 0:
            notes = detect_money(image, [tr.face_box for tr in people if tr.face_box])
            self._money_step(notes, t)
        self._last_notes = notes if self.count_money and self.frames_analysed % MONEY_EVERY == 0 else getattr(self, "_last_notes", [])
        states = {}
        for tr in current:
            if tr.label == "person":
                tr.activity = activity.classify(tr.box, devices)
                if tr.status == "named":
                    states[tr.name] = tr.activity
                    if tr.attrs.mood:
                        self.activity.mood(tr.name, tr.attrs.mood)
        with self._lock:
            self.activity.update(t, states)
        self._log(image, current, t, lists)
        return self._draw(image, current, t, lists, devices, self._last_notes)

    def _money_step(self, notes, t):
        """A note can be missed for a frame (a hand, glare), so the count is the highest total of the
        last few checks; it is logged once it has held for MONEY_STEADY checks."""
        total = money_total(notes)
        self._money_recent = (self._money_recent + [(total, notes)])[-(MONEY_STEADY + 2):]
        best_total, best_notes = max(self._money_recent, key=lambda x: x[0])
        history = self._money_history = (getattr(self, "_money_history", []) + [best_total])[-3 * MONEY_STEADY:]
        last = self.money["events"][-1]["total"] if self.money["events"] else 0
        # more money shows up quickly; less money must hold 3x longer (a missed note is not cash leaving)
        need = MONEY_STEADY if best_total >= last else 3 * MONEY_STEADY
        steady = len(history) >= need and len(set(history[-need:])) == 1
        with self._lock:
            self.money["now"] = best_total
            if steady:
                self.money["max"] = max(self.money["max"], best_total)
                if best_total != last and best_total > 0:
                    self.money["events"].append({"time": self.clock(t), "seconds": round(t, 1),
                                                 "total": best_total, "notes": money_breakdown(best_notes)})
                    self.money["notes"] = money_breakdown(best_notes)

    def demographics(self):
        ages, moods = {}, {}
        for age, mood, sentiment in self.person_attrs.values():
            if age:
                ages[age] = ages.get(age, 0) + 1
            if sentiment:
                moods[sentiment] = moods.get(sentiment, 0) + 1
        return {"ages": [[g, ages.get(g, 0)] for g in face_attributes.AGE_GROUPS if ages.get(g)],
                "sentiment": moods, "people": len(self.person_attrs)}

    def _keep_sample(self, image, tr, face_box):
        x1, y1, x2, y2 = face_box
        fw, fh = x2 - x1, y2 - y1
        if fw < MIN_SAMPLE_FACE:
            return
        h, w = image.shape[:2]
        mx, my = fw * 0.6, fh * 0.6
        crop = image[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
        if crop.size == 0:
            return
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        quality = fw * min(sharp, 400) ** 0.5
        tr.samples.append((quality, crop.copy()))
        tr.samples = sorted(tr.samples, key=lambda q: -q[0])[:SAMPLES_PER_PERSON]
        with self._lock:
            self.samples[tr.id] = tr.samples
            self.tracks[tr.id] = tr

    def acknowledge(self, track_id, by=""):
        """The operator saw the alarm for this person and is taking action."""
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            for e in self.events + self.vehicle_log:
                if e["track"] == track_id and e.get("alert") and not e.get("ack"):
                    e["ack"] = stamp + (f" by {by}" if by else "")
        return stamp

    def add_person(self, track_id, name, list_name):
        """'This unknown person is <name>': save their CCTV face photos and put them on a list."""
        name = faces.clean_name(name)
        if not name:
            raise ValueError("Type a name.")
        if list_name not in LIST_NAMES:
            raise ValueError("Choose green, red or black.")
        with self._lock:
            samples = list(self.samples.get(track_id, []))
        if not samples:
            raise ValueError("No clear face of this person was captured, so they can't be added.")
        folder = faces.person_folder(name)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = 0
        for i, (_, crop) in enumerate(samples):
            scale = max(1.0, 240 / crop.shape[1])      # enlarge small CCTV faces
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if len(faces.detect_faces(big)) == 0:      # the photo must still contain a findable face
                continue
            cv2.imwrite(os.path.join(folder, f"cctv_{stamp}_{i}.jpg"), big)
            saved += 1
        if not saved:
            if not os.listdir(folder):
                os.rmdir(folder)
            raise ValueError("The captured face is too small or blurry to save.")
        faces.build_face_db()
        set_list(name, list_name)
        with self._lock:
            for e in self.events:
                if e["track"] == track_id:
                    e.update(name=name, list=list_name, alert=list_name in ("red", "black"))
            tr = self.tracks.get(track_id)
            if tr is not None:
                tr.name, tr.status, tr.announced = name, "named", "named"
        return {"name": name, "list": list_name, "photos": saved}

    def _log(self, image, current, t, lists):
        plate_lists = lpr.load_lists() if self.read_plates else {}
        with self._lock:
            self.people_seen = sum(1 for tr in self._all_ids(current) if tr == "person")
            for tr in current:
                if tr.label in VEHICLES or tr.status == "pending" or tr.announced == tr.status:
                    continue
                tr.announced = tr.status     # unknown -> named later gives a second, corrected entry
                group = list_of(tr.name, lists)
                self.events.append({
                    "time": clock(t) if self.is_file else time.strftime("%H:%M:%S"),
                    "seconds": round(t, 1), "track": tr.id,
                    "name": tr.name or "Unknown person", "list": group,
                    "alert": group in ("red", "black"),
                    "photo": thumbnail(image, tr.face_box or tr.box),
                    "can_add": bool(tr.samples),
                    "age": tr.attrs.age_group, "mood": tr.attrs.mood, "sentiment": tr.attrs.sentiment,
                })
            for tr in current:
                if tr.label in VEHICLES and not tr.announced and tr.seen >= VEHICLE_FRAMES:
                    tr.announced = "vehicle"
                    self.vehicles_seen += 1
                    self.vehicle_log.append({"time": self.clock(tr.first), "seconds": round(tr.first, 1),
                                             "track": tr.id, "type": tr.label, "colour": tr.colour,
                                             "photo": tr.snapshot[1] if tr.snapshot else None,
                                             "plate": None, "list": "unknown", "alert": False})
                if tr.label in VEHICLES and tr.announced == "vehicle":
                    for v in self.vehicle_log:          # colour, photo and plate improve while it is in view
                        if v["track"] == tr.id:
                            v["colour"], v["type"] = tr.colour, tr.label
                            if tr.snapshot:
                                v["photo"] = tr.snapshot[1]
                            plate = tr.plates.result
                            if plate and v.get("key") != plate["key"]:
                                entry = lpr.lookup(plate["key"], plate_lists)
                                ok, buf = cv2.imencode(".jpg", plate["crop"])
                                if self.collect_plates:
                                    lpr.save_for_training(plate["crop"], plate["plate"])
                                v.update(plate=plate["plate"], latin=plate["latin"], key=plate["key"],
                                         list=entry["list"], owner=entry.get("owner", ""),
                                         alert=entry["list"] in ("red", "black"),
                                         plate_photo="data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None)
                            elif plate:
                                entry = lpr.lookup(plate["key"], plate_lists)
                                if entry["list"] != v.get("list"):       # list changed while it is in view
                                    v.update(list=entry["list"], owner=entry.get("owner", ""),
                                             alert=entry["list"] in ("red", "black"))

    def _all_ids(self, current):
        if not hasattr(self, "_seen"):
            self._seen = {}
        for tr in current:
            if tr.seen >= VEHICLE_FRAMES or tr.status != "pending":   # ignore one-frame false detections
                self._seen[tr.id] = "person" if tr.label == "person" else "vehicle"
        return self._seen.values()

    def clock(self, t):
        """Video time for recordings, wall-clock time for live cameras."""
        return clock(t) if self.is_file else time.strftime("%H:%M:%S", time.localtime(self.wall_start + t))

    def log_csv(self):
        """Everyone and every vehicle seen, with time, list, age group, mood and plate."""
        import csv, io
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["Time", "Kind", "Name / vehicle", "List", "Age group", "Mood", "Plate", "Plate (Latin)",
                    "Owner", "Acknowledged"])
        with self._lock:
            rows = [(e["seconds"], [e["time"], "person", e["name"], e["list"], e.get("age") or "", e.get("mood") or "",
                                    "", "", "", e.get("ack") or ""]) for e in self.events]
            rows += [(v["seconds"], [v["time"], "vehicle", f'{v.get("colour") or ""} {v["type"]}'.strip(), v.get("list", ""),
                                     "", "", v.get("plate") or "", v.get("latin") or "", v.get("owner") or "",
                                     v.get("ack") or ""]) for v in self.vehicle_log]
        for _, r in sorted(rows, key=lambda x: x[0]):
            w.writerow(r)
        return out.getvalue()

    def report_csv(self):
        with self._lock:
            return self.activity.csv(self.clock)

    def _draw(self, image, current, t, lists, devices=(), notes=()):
        out = image.copy()
        thick = max(2, image.shape[1] // 500)
        font = 0.5 + image.shape[1] / 2600
        for label, box, conf in notes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (131, 212, 70), thick)
            value = i18n.CURRENCY_VALUE.get(label, 0)
            cv2.putText(out, f"{value:g} EGP", (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (131, 212, 70), 2, cv2.LINE_AA)
        for label, box in devices:                 # thin outline on computers and phones
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200) if label != activity.PHONE else (60, 106, 255), 1)
        for tr in current:
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            if tr.label in VEHICLES:
                colour, text = COLOURS["vehicle"], (f"{tr.colour} {tr.label}" if tr.colour else tr.label)
                plate = tr.plates.result
                if plate:
                    group = lpr.lookup(plate["key"]).get("list", "unknown")
                    colour = COLOURS.get(group, COLOURS["vehicle"]) if group != "unknown" else COLOURS["vehicle"]
                    text += f"  {plate['latin']}"
                if tr.plate_box is not None:
                    px1, py1, px2, py2 = [int(v) for v in tr.plate_box]
                    cv2.rectangle(out, (px1, py1), (px2, py2), (0, 194, 242), 2)
            elif tr.status == "named":
                group = list_of(tr.name, lists)
                colour, text = COLOURS[group], f"{tr.name}  [{group}]"
                extra = {"computer": "at computer", "phone": "on phone"}.get(getattr(tr, "activity", ""), "")
                if extra:
                    text += f"  {extra}"
            elif tr.status == "unknown":
                colour, text = COLOURS["unknown"], "Unknown"
                if tr.attrs.age_group:
                    text += f"  ~{tr.attrs.age_group}"
            else:
                colour, text = COLOURS["pending"], "Person"
            if tr.label == "person" and tr.attrs.mood and tr.attrs.mood != "calm":
                text += f"  {tr.attrs.mood}"
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
            if tr.face_box is not None and tr.label == "person":
                fx1, fy1, fx2, fy2 = [int(v) for v in tr.face_box]
                cv2.rectangle(out, (fx1, fy1), (fx2, fy2), colour, 1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            ty = max(th + 6, y1)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty), colour, -1)
            dark = colour in (COLOURS["vehicle"], COLOURS["unknown"], COLOURS["pending"])
            cv2.putText(out, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (20, 20, 20) if dark else (255, 255, 255), 1, cv2.LINE_AA)
        if notes:
            cv2.putText(out, f"Cash: {self.money['now']:g} EGP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font * 1.2,
                        (131, 212, 70), 2, cv2.LINE_AA)
        stamp = clock(t) if self.is_file else time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, stamp, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, font,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out


class LiveReader:
    """Keeps only the newest frame of a live stream, so the AI never falls behind."""
    def __init__(self, cap):
        self.cap, self.frame, self.dead = cap, None, False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.failures = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.failures += 1
                if self.failures > 100:
                    self.dead = True
                    return
                time.sleep(0.05)
                continue
            self.failures = 0
            with self._lock:
                self.frame = frame

    def latest(self):
        with self._lock:
            frame, self.frame = self.frame, None
        return frame

    def close(self):
        self._stop.set()


# ---------------------------------------
# Job registry
# ---------------------------------------
JOBS = {}


def start(source, title, is_file, count_money=False):
    for job in list(JOBS.values()):          # one job at a time: the laptop has one CPU/GPU
        job.stop()
    job = CCTVJob(source, title, is_file, count_money)
    JOBS[job.id] = job
    return job


def get(job_id):
    return JOBS.get(job_id)
