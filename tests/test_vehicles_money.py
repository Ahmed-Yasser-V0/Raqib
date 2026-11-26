"""Money, tracking and vehicles."""
import cv2
import numpy as np

import cctv


def test_money_on_the_test_photo():
    image = cctv.shrink(cv2.imread("images/all.jpg"))
    notes = cctv.detect_money(image)
    assert cctv.money_total(notes) >= 700        # 770 EGP on the photo
    assert "×" in cctv.money_breakdown(notes)


def test_money_ignored_on_faces():
    notes = cctv.detect_money(cctv.shrink(cv2.imread("images/all.jpg")), [[0, 0, 10000, 10000]])
    assert notes == []


def test_tracker_follows_fast_mover():
    tr = cctv.Tracker()
    a = tr.step([("person", [100, 100, 130, 180], 0.9)], 0.0)[0]
    b = tr.step([("person", [140, 100, 170, 180], 0.9)], 0.25)[0]   # moved more than its width
    assert a is b


def test_vehicle_colours():
    white = np.full((120, 200, 3), 235, np.uint8)
    red = np.zeros((120, 200, 3), np.uint8); red[:] = (30, 30, 200)
    black = np.full((120, 200, 3), 25, np.uint8)
    assert cctv.vehicle_colour(white, [0, 0, 200, 120]) == "white"
    assert cctv.vehicle_colour(red, [0, 0, 200, 120]) == "red"
    assert cctv.vehicle_colour(black, [0, 0, 200, 120]) == "black"


def test_vehicle_type_vote():
    tr = cctv.Tracker()
    a = tr.step([("car", [0, 0, 100, 50], 0.9)], 0)[0]
    for i in range(1, 4):
        tr.step([("bus", [0, 0, 100, 50], 0.6)], i * 0.25)
    tr.step([("car", [0, 0, 100, 50], 0.9)], 1.25)
    assert a.label == "bus"          # majority wins (3 bus vs 2 car)
