"""Egyptian plate reading and car lists."""
import cv2
import pytest

import lpr


def test_text_and_key():
    assert lpr.text("سطن", "٧١٢٦") == "س ط ن ٧١٢٦"
    assert lpr.key("س ط ن ٧١٢٦") == lpr.key("سطن 7126") == "سطن7126"
    assert lpr.normalise("سطن 7126") == "س ط ن ٧١٢٦"
    assert lpr.latin("سطن", "٧١٢٦") == "S T N 7126"


def test_valid_plates():
    assert lpr.valid("سطن", "٧١٢٦") and lpr.valid("د", "١")
    assert not lpr.valid("", "١٢٣") and not lpr.valid("سطنم", "١") and not lpr.valid("س", "١٢٣٤٥")


def test_votes_need_two_reads():
    v = lpr.PlateVotes()
    r = {"key": "سطن7126", "conf": 0.8, "plate": "س ط ن ٧١٢٦"}
    v.add(r)
    assert v.result is None or v.votes["سطن7126"] >= 2
    v.add(dict(r)); v.add({"key": "سطم7126", "conf": 0.5, "plate": "x"})
    assert v.result["key"] == "سطن7126"


def test_car_lists(empty_faces):
    lpr.set_list("سطن 7126", "black", "Unknown van")
    assert lpr.lookup(lpr.key("س ط ن ٧١٢٦"))["list"] == "black"
    with pytest.raises(ValueError):
        lpr.set_list("س", "purple")
    lpr.remove("س ط ن ٧١٢٦")
    assert lpr.lookup(lpr.key("سطن7126"))["list"] == "unknown"


def test_plates_route(client, empty_faces):
    r = client.post("/cctv/plates", data={"plate": "ب ط ل 5555", "list": "green", "owner": "Manager"})
    assert r.status_code == 200
    plates = r.get_json()["plates"]
    assert plates[0]["plate"] == "ب ط ل ٥٥٥٥" and plates[0]["owner"] == "Manager"
    assert client.post("/cctv/plates", data={"plate": "x", "list": "green"}).status_code == 400


@pytest.mark.skipif(not lpr.available(), reason="plate models not installed")
def test_reads_a_real_plate_crop():
    import glob
    crops = sorted(glob.glob("tests/data/plate_*.png"))
    if not crops:
        pytest.skip("no sample plates")
    for c in crops:
        letters, digits, _ = lpr.read_characters(cv2.imread(c))
        expected = c.rsplit("_", 1)[1].rsplit(".", 1)[0]
        assert lpr.key(lpr.text(letters, digits)) == expected
