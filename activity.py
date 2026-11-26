# ---------------------------------------
# Working-time tracker for the CCTV console
# ---------------------------------------
# For every recognised person it keeps, from what the camera sees:
#   computer  - next to a laptop / monitor / keyboard / mouse (working)
#   phone     - a mobile phone in the hand or at the ear
#   present   - in view but not at a computer
#   away      - not seen; longer than BREAK_AFTER seconds counts as a break
#
# Honest limits: the camera cannot read a screen, so "computer" means "sitting
# at a computer", not "doing work on it". A phone that is hidden by the hand or
# the desk is not seen. People are only counted once their face has been
# recognised; someone who never shows their face stays "unknown".
# ---------------------------------------
import csv
import io

DEVICES = {"laptop", "tv", "keyboard", "mouse"}       # "tv" = a computer monitor in COCO
PHONE = "cell phone"
BREAK_AFTER = 60.0          # seconds out of view before it counts as a break
SMOOTH = 3                  # frames voted over, so one missed phone does not flip the state
MAX_STEP = 5.0              # never count more than this between two analysed frames

STATE_TEXT = {"computer": "Working at computer", "phone": "On phone",
              "present": "Away from computer", "away": "On break"}


def overlap(a, b):
    """Intersection area divided by the smaller box (1 = one box is inside the other)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def grow(box, fx, fy):
    w, h = box[2] - box[0], box[3] - box[1]
    return [box[0] - w * fx, box[1] - h * fy, box[2] + w * fx, box[3] + h * fy]


def classify(person_box, devices):
    """What one person is doing in one frame. devices: [(label, box)]."""
    x1, y1, x2, y2 = person_box
    upper = [x1, y1, x2, y1 + (y2 - y1) * 0.75]          # hands and head, not the feet
    for label, box in devices:
        if label == PHONE and overlap(upper, box) > 0.5:
            return "phone"
    reach = grow(person_box, 0.35, 0.10)                  # an arm's length to the sides
    for label, box in devices:
        if label in DEVICES and overlap(reach, box) > 0.3:
            return "computer"
    return "present"


def duration(seconds):
    seconds = int(round(seconds))
    h, m = seconds // 3600, seconds % 3600 // 60
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {seconds % 60:02d}s"
    return f"{seconds}s"


class Person:
    def __init__(self, name, t):
        self.name = name
        self.first = self.last = t
        self.totals = {"computer": 0.0, "phone": 0.0, "present": 0.0}
        self.breaks = []             # [[start, end]]
        self.segments = []           # [[start, end, state]] for the timeline bar
        self.recent = []
        self.state = "present"
        self.moods = {}

    def see(self, state, t):
        if t - self.last > BREAK_AFTER:
            self.breaks.append([self.last, t])
        step = min(MAX_STEP, max(0.0, t - self.last))
        self.recent = (self.recent + [state])[-SMOOTH:]
        smoothed = max(set(self.recent), key=self.recent.count)
        if step and t - self.last <= BREAK_AFTER:
            self.totals[self.state] += step          # time since the last frame goes to the last state
            self._segment(self.last, t, self.state)
        self.state, self.last = smoothed, t

    def _segment(self, start, end, state):
        if self.segments and self.segments[-1][2] == state and start - self.segments[-1][1] <= MAX_STEP:
            self.segments[-1][1] = end
        else:
            self.segments.append([start, end, state])

    def current(self, now):
        return "away" if now - self.last > BREAK_AFTER else self.state

    def summary(self, now, clock):
        state = self.current(now)
        breaks = list(self.breaks)
        if state == "away":
            breaks.append([self.last, None])          # still on break
        break_time = sum((e if e is not None else now) - s for s, e in breaks)
        return {
            "name": self.name, "state": state, "state_text": STATE_TEXT[state],
            "since": clock(self.last) if state == "away" else None,
            "first": clock(self.first), "last": clock(self.last),
            "computer": round(self.totals["computer"]), "phone": round(self.totals["phone"]),
            "present": round(self.totals["present"]), "break_time": round(break_time),
            "breaks": [{"from": clock(s), "to": clock(e) if e is not None else None,
                        "minutes": round(((e if e is not None else now) - s) / 60, 1)} for s, e in breaks],
            "moods": [[m, round(100 * n / sum(self.moods.values()))]
                      for m, n in sorted(self.moods.items(), key=lambda kv: -kv[1])] if self.moods else [],
            "segments": [[round(s, 1), round(e, 1), st] for s, e, st in self.segments[-400:]],
            "text": {"computer": duration(self.totals["computer"]), "phone": duration(self.totals["phone"]),
                     "present": duration(self.totals["present"]), "break": duration(break_time)},
        }


class ActivityLog:
    def __init__(self):
        self.people = {}
        self.start = None
        self.now = 0.0

    def update(self, t, people_states):
        """people_states: {name: state} for recognised people seen in this frame."""
        if self.start is None:
            self.start = t
        self.now = t
        for name, state in people_states.items():
            if name not in self.people:
                self.people[name] = Person(name, t)
            self.people[name].see(state, t)

    def mood(self, name, mood):
        if name in self.people:
            m = self.people[name].moods
            m[mood] = m.get(mood, 0) + 1

    def rename(self, old, new):
        if old in self.people and new not in self.people:
            self.people[new] = self.people.pop(old)
            self.people[new].name = new

    def summary(self, clock):
        span = (self.now - self.start) if self.start is not None else 0
        return {"start": 0 if self.start is None else round(self.start, 1), "now": round(self.now, 1),
                "span": round(span, 1),
                "people": [p.summary(self.now, clock) for p in
                           sorted(self.people.values(), key=lambda p: p.first)]}

    def csv(self, clock):
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["Name", "First seen", "Last seen", "Working at computer", "On phone",
                    "Away from computer (in view)", "Breaks", "Break time", "Break times", "Mood", "Now"])
        for p in self.summary(clock)["people"]:
            w.writerow([p["name"], p["first"], p["last"], p["text"]["computer"], p["text"]["phone"],
                        p["text"]["present"], len(p["breaks"]), p["text"]["break"],
                        "; ".join(f'{b["from"]}-{b["to"] or "now"}' for b in p["breaks"]),
                        ", ".join(f"{m} {pc}%" for m, pc in p["moods"]), p["state_text"]])
        return out.getvalue()
