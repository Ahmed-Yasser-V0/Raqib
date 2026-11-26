# Raqib (رقيب)

*Raqib* means "watcher" in Arabic. It is CCTV analytics for companies: who came in, when, and whether they are allowed.
It grew out of the face recognition in [Ru'ya](https://github.com/Ahmed-Yasser-V0/Ruya), a smart-glasses graduation project for blind and visually impaired people.

## What it does
- **People**: face recognition with **green / red / black** lists, a full-screen alarm for black-listed people, a Seen log with time, age group and mood.
- **Cars**: Egyptian plate reading (trained on EALPR), car lists by plate, vehicle type and colour, and an alarm for black-listed cars.
- **Working time**: time at the computer, on the phone and on breaks per recognised person, with an Excel report.
- **Money**: counts Egyptian banknotes on a counter camera.
- **Sources**: a recording (MP4, AVI, MKV, DVR export, up to 2 GB) or a live RTSP camera (Hikvision, Dahua) or a webcam.

## Run it (Windows)
1. `setup.bat` (once), then `run.bat`.
2. Open **http://localhost:5002** in Chrome.
3. Add people under **Lists → People** (name, list, 1–5 face photos) and cars under **Lists → Cars**.

Raqib listens on this computer only (`127.0.0.1`). To open it from another PC, set `RAQIB_HOST=0.0.0.0` **only after adding a login**: the page shows cameras, faces and the black list.

## Models and accuracy
| Part | Model | Accuracy |
|---|---|---|
| Faces | OpenCV YuNet + SFace | 90% named on real test faces with 3 photos, 0 wrong names |
| Mood | OpenCV Zoo MobileFaceNet | 88.3% on RAF-DB (published) |
| Age group | GoogLeNet (Adience) | ~50% exact, ~85% within one group (published); research-only data |
| People / vehicles | YOLOv8m | 50.2 mAP50-95 on COCO (published) |
| Plate finder | YOLO11n trained on EALPR | 99.4% mAP50 on 260 unseen cars |
| Plate reader | YOLO11n trained on EALPR | 93.9% of unseen plates exactly right (one frame); 13 of 14 in video |
| Money | custom YOLOv8 (13 notes) | 770 EGP exactly right on the test photo; no plastic 20 note yet |

## Before selling
- **Licences**: Ultralytics YOLO is AGPL-3.0 (a commercial licence is needed for a closed product); EALPR must be credited; the Adience age data is research-only.
- **Law**: faces and employee monitoring are sensitive personal data under Egypt's Personal Data Protection Law 151/2020: tell staff, keep a written policy, limit who can see the page.
- **Gates**: for real gates, Hikvision ANPR cameras read plates inside the camera; Raqib can take their plate events instead of its own plate reader.

## Tests
`.venv\Scripts\python.exe -m pytest tests`

## Credits
- Plate data: **EALPR** (Youssef et al., *Real-time Egyptian License Plate Detection and Recognition using YOLO*, IJACSA 2022), CC BY 4.0.
- Sample videos: Intel IoT DevKit sample videos (CC BY 4.0).
- Fonts: Atkinson Hyperlegible (Braille Institute) and IBM Plex Sans Arabic, SIL Open Font Licence.

## History
### First version: CCTV console
Open **http://localhost:5002**.

- **Recording**: drop a video file (MP4, AVI, MKV, MOV, DVR export, up to 2 GB). The video plays with boxes, and you can download the analysed copy when it finishes.
- **Live camera**: type an RTSP address, for example
  - Hikvision `rtsp://user:password@192.168.1.64:554/Streaming/Channels/101`
  - Dahua `rtsp://user:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
  - or `0` for the laptop webcam.
- People added under Lists → People (or from the video) are recognised. Each person is followed across frames and needs **2 sure matches** before a name is shown. After 6 clear faces with no match they are logged as **unknown**.
- **Lists**: each known person can be put on **green** (allowed), **red** (watch) or **black** (banned). Red and black raise an alert with a beep. Lists are saved in `faces/lists.json`.
- CCTV faces are small, so faces are searched inside each person box and enlarged ×2–3 first.
- Tested on a public Intel sample corridor video (768×432): 1 enrolled person was named and put on the watch list correctly, and 6 strangers were logged as unknown, with 0 wrong names.

### v2.3 — Name unknown people, working time, black-list alarm
- **Add to a list**: every unknown person in *Seen* has an **Add to a list** button. Type a name and choose green, red or black. Their best face photos from the video are saved, and they are recognised from then on (tested: an unknown person was named correctly on the next run).
- **Working time** for each recognised person: time *at computer* (next to a laptop, monitor, keyboard or mouse), *on phone* (phone in the hand or at the ear), *in view but not at a computer*, and *breaks* (out of view for more than 60 s). There is a timeline bar per person and a **Download report (Excel)** CSV.
  - Honest limits: the camera can't see the screen, so "at computer" means sitting at it, not proof of work. A hidden phone isn't seen. Only recognised people are counted.
  - Monitoring employees counts as processing personal data under Egypt's Personal Data Protection Law 151/2020. Tell staff and keep a written policy before using it for real.
- **Black-list alarm**: a full-screen red alarm with a large photo, a siren and a spoken warning repeated every 10 s until someone presses **Acknowledge**. It also sends a Windows notification when the browser is in the background, and records who acknowledged it and when. A red (watch) list person gets a banner, a chime and one spoken warning.

### v2.4 — Mood (sentiment), age group, money and counts
- **Mood**: OpenCV Zoo facial-expression model (Apache-2.0, 88.3% on RAF-DB). Seven expressions are grouped as positive, calm or negative.
  - Glasses: "Ahmed is at 12 o'clock, about 1 meter away, and looks happy". Calm faces are not announced. Groups: "Eight unknown people, most of them look happy".
  - CCTV: mood on each person's label, in the Seen list, as a mood mix per worker, and as a positive/calm/negative bar.
  - Tested: 7 of 8 smiling people in a team photo were read as happy.
- **Age group**: GoogLeNet trained on Adience (ONNX Model Zoo). 8 groups. Published accuracy is ~50% for the exact group and ~85% within one neighbouring group, so it is labelled "approximate". It is voted over frames and only given for faces ≥ 40 px (CCTV) or ≥ 60 px (glasses). Adience images were collected for research, so check the licence before selling the age feature.
- **Count money** switch on the CCTV page finds Egyptian notes, draws their values and keeps a log of totals. Tested on the 12-note photo: 770 EGP, exactly right. A note missed for a moment does not lower the logged total.
- **People counts**: in view now, the most at once, and the total. The tracker was improved for fast movers: on the street sample, 18 people (over-counted) became 4 (correct).
- Samples in `samples/`: `corridor_sample.mp4` (faces), `parking_cars.mp4`, `street_mixed.mp4` (Intel sample videos, CC-BY 4.0), and `cash_counter.mp4`.

### v2.5 — Vehicle colour and vehicle log
- Each car, truck, bus, motorcycle and bicycle is logged once, with its type, **colour** (white, silver, grey, black, red, blue, green, yellow, orange, brown), the time and its best snapshot. They appear in the **Vehicles** list and on the video label ("white car").
- Colour ignores windows, tyres and shadows (the darkest 35% of pixels). A slight tint counts as silver or metallic. The colour is voted over frames.
- The vehicle type is also voted over frames. From a high overhead angle YOLO may still call a car a "bus" (1 of the 4 cars in the parking sample).
- People and vehicles are counted only after being seen in 3 frames, which removes one-frame false detections.
- Make and model are not recognised. For gates, plan Egyptian plate reading: YOLO to find the plate, then YOLO to read each character (EALPR dataset, CC BY 4.0), or Hikvision ANPR cameras.

### v2.6 — Egyptian plate reading (LPR) and car lists
- `lpr.py` with two small YOLO11n models trained on the public **EALPR** dataset (Youssef et al., 2022). Please cite it.
  - `models/plate_detector.pt` finds the plate: **mAP50 99.4%** on 260 held-out car photos.
  - `models/plate_chars.pt` reads the 17 letters and 10 digits: **93.9% of unseen plates read exactly right** from a single frame (232/247).
  - Video test (`samples/egypt_plates.mp4`, cars held out from both models): **13 of 14 different plates exactly right**, with 1 wrong letter. The black-listed car raised the alarm.
- Letters are read right to left and digits left to right. Plates are shown in Arabic (س ى د ٩٩٩) and Latin (S Y D 999). The reading is voted over frames, and a different plate read twice in the same spot counts as a new car (a queue at a gate).
- **Car lists** (Lists → Cars): add a plate with an owner and green/red/black, or press "Add to a list" on a car the camera read. A black-listed car raises the full alarm; red raises the banner.
- **New letters such as ك**: the EALPR plates don't contain ك. Every plate the CCTV reads is saved in `training/plates_new/` with the model's labels. Fix the labels on 30–50 plates that have ك (class 27), then run `training\train_plate_chars.bat`. It merges them with EALPR, retrains, and only installs the new model if it scores at least 95%.
- **Download log (Excel)**: every person and vehicle seen, with list, age group, mood, plate, owner and acknowledgement.
