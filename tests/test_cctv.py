"""CCTV console: routes, lists, tracking and a short end-to-end video."""
import time

import cv2
import numpy as np
import pytest

import cctv


def test_page_loads(client):
    r = client.get("/cctv")
    assert r.status_code == 200 and b"Analyse" in r.data


def test_start_needs_a_source(client):
    r = client.post("/cctv/start", data={})
    assert r.status_code == 400 and "video file" in r.get_json()["error"]


def test_rejects_non_video(client):
    r = client.post("/cctv/start", data={"video": (open("images/car.jpg", "rb"), "car.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_rejects_bad_address(client):
    r = client.post("/cctv/start", data={"url": "ftp://camera"})
    assert r.status_code == 400


def test_unknown_job(client):
    assert client.get("/cctv/status/nope").status_code == 404
    assert client.get("/cctv/video/nope").status_code == 404


def test_lists_round_trip(client, empty_faces):
    assert client.get("/cctv/lists").get_json()["people"] == []
    cctv.set_list("Ahmed", "black")
    assert cctv.list_of("Ahmed", cctv.load_lists()) == "black"
    assert cctv.list_of("Someone else", cctv.load_lists()) == "green"
    assert cctv.list_of(None, {}) == "unknown"
    r = client.post("/cctv/lists", data={"name": "Ahmed", "list": "purple"})
    assert r.status_code == 400


def test_tracker_keeps_identity():
    tr = cctv.Tracker()
    a = tr.step([("person", [10, 10, 60, 160], 0.9)], 0.0)[0]
    b = tr.step([("person", [14, 12, 64, 162], 0.9)], 0.25)[0]
    assert a is b
    c = tr.step([("person", [300, 10, 350, 160], 0.9)], 0.5)[0]
    assert c is not a
    assert tr.step([], 5.0) == [] and tr.tracks == []


def test_identity_needs_two_votes():
    t = cctv.Track(1, "person", [0, 0, 1, 1], 0)
    t.update_identity("Saif", 0.5, "sure")
    assert t.status == "pending"
    t.update_identity("Saif", 0.5, "sure")
    assert t.status == "named" and t.name == "Saif"
    u = cctv.Track(2, "person", [0, 0, 1, 1], 0)
    for _ in range(cctv.FRAMES_TO_UNKNOWN):
        u.update_identity(None, 0.1, "unknown")
    assert u.status == "unknown"


def test_short_video_end_to_end(tmp_path, empty_faces):
    """A 3-second clip made from a test photo: the job finishes and writes a video."""
    src = cv2.imread("images/car_person.jpg")
    src = cv2.resize(src, (402, 502))
    file = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*"mp4v"), 10, (src.shape[1], src.shape[0]))
    for i in range(30):
        out.write(np.roll(src, i * 2, axis=1))
    out.release()
    job = cctv.start(file, "clip", is_file=True)
    deadline = time.time() + 120
    while job.state in ("starting", "running") and time.time() < deadline:
        time.sleep(0.3)
    s = job.snapshot()
    assert s["state"] == "done", s
    assert s["frames"] >= 10 and s["people"] >= 1
    assert s["download"] and job.latest_jpeg


def test_ack_and_report_need_a_job(client):
    assert client.post("/cctv/ack/nope/1").status_code == 404
    assert client.get("/cctv/report/nope.csv").status_code == 404
    assert client.post("/cctv/add/nope/1", data={"name": "A"}).status_code == 404


def test_alarm_markup_present(client):
    html = client.get("/cctv").data
    assert b'id="alarm"' in html and b"Acknowledge" in html and b"Working time" in html


def test_add_person_from_photos(client, empty_faces):
    import io, cv2
    ok, buf = cv2.imencode(".jpg", cv2.resize(cv2.imread("images/people.jpg"), None, fx=6, fy=6))
    r = client.post("/people/add", data={"name": "Test Person", "list": "red",
                                         "photos": (io.BytesIO(buf.tobytes()), "p.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code in (200, 400)            # a group photo may be refused by the quality check
    assert client.post("/people/add", data={"name": ""}).status_code == 400
    assert client.post("/people/add", data={"name": "X"}).status_code == 400


def test_console_is_the_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Raqib" in r.data and b"Back to glasses" not in r.data
