"""Working-time tracker: what a person is doing and how long."""
import activity as A

PERSON = [100, 50, 300, 450]


def test_phone_in_hand():
    assert A.classify(PERSON, [("cell phone", [180, 200, 210, 250])]) == "phone"


def test_phone_on_the_floor_is_not_use():
    assert A.classify(PERSON, [("cell phone", [180, 430, 210, 470])]) != "phone"


def test_at_computer():
    assert A.classify(PERSON, [("laptop", [290, 250, 420, 330])]) == "computer"
    assert A.classify(PERSON, [("tv", [600, 50, 800, 200])]) == "present"


def test_phone_beats_computer():
    devices = [("laptop", [290, 250, 420, 330]), ("cell phone", [150, 100, 180, 150])]
    assert A.classify(PERSON, devices) == "phone"


def test_totals_and_breaks():
    log = A.ActivityLog()
    t = 0.0
    for _ in range(40):                     # 10 s at the computer (4 frames a second)
        log.update(t, {"Ahmed": "computer"}); t += 0.25
    for _ in range(20):                     # 5 s on the phone
        log.update(t, {"Ahmed": "phone"}); t += 0.25
    t += 120                                # away for two minutes -> a break
    for _ in range(8):
        log.update(t, {"Ahmed": "computer"}); t += 0.25
    p = log.summary(A.duration if False else (lambda s: f"{s:.0f}"))["people"][0]
    assert 9 <= p["computer"] <= 12 and 4 <= p["phone"] <= 6
    assert len(p["breaks"]) == 1 and p["break_time"] >= 120
    assert p["state"] == "computer"
    assert "Ahmed" in log.csv(lambda s: f"{s:.0f}")


def test_current_break():
    log = A.ActivityLog()
    log.update(0, {"Saif": "present"})
    log.update(1, {"Other": "present"})
    log.update(200, {"Other": "present"})
    saif = [p for p in log.summary(str)["people"] if p["name"] == "Saif"][0]
    assert saif["state"] == "away" and saif["breaks"][-1]["to"] is None


def test_one_missed_frame_does_not_flip():
    log = A.ActivityLog()
    for i, st in enumerate(["computer"] * 5 + ["present"] + ["computer"] * 3):
        log.update(i * 0.25, {"A": st})
    assert log.people["A"].state == "computer"
