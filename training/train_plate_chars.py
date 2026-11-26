"""Retrain the plate-character reader, adding new letters (for example ك).

1. The CCTV page saves every plate it reads into training/plates_new/images + labels
   (labels are the model's own guesses).
2. Open those folders in a labelling tool (LabelImg, or upload to Roboflow), fix the plates that
   have the new letter, and give it class number 27 (= ك, see NEW_LETTERS below).
   Aim for 30-50 plates that contain each new letter.
3. Run train_plate_chars.bat. It downloads the original EALPR plates (first time only, ~600 MB),
   merges your plates (repeated 5x so the new letter is learned), trains, and only replaces
   models/plate_chars.pt if the new model reads the old test plates at least as well.
"""
import glob
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EALPR = os.path.join(HERE, "datasets", "EALPR")
NEW = os.path.join(HERE, "plates_new")
WORK = os.path.join(HERE, "plate_chars_work")
OLD = ["أ", "ب", "ج", "د", "ر", "س", "ص", "ط", "ع", "ف", "ق", "ل", "م", "ن", "ھ", "و", "ى",
       "٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
NEW_LETTERS = ["ك"]                 # class 27; add more here if needed (28, 29, ...)
NAMES = OLD + NEW_LETTERS


def get_ealpr():
    if not os.path.isdir(EALPR):
        os.makedirs(os.path.dirname(EALPR), exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ahmedramadan96/EALPR.git", EALPR])


def build():
    random.seed(7)
    shutil.rmtree(WORK, ignore_errors=True)
    pairs = []
    for img in glob.glob(os.path.join(EALPR, "EALPR- Plates dataset", "*.png")):
        lab = os.path.join(EALPR, "EALPR- LP characters dataset", "Characters Labeling",
                           os.path.splitext(os.path.basename(img))[0] + ".txt")
        if os.path.exists(lab):
            pairs.append((img, lab, 1))
    for img in glob.glob(os.path.join(NEW, "images", "*")):
        lab = os.path.join(NEW, "labels", os.path.splitext(os.path.basename(img))[0] + ".txt")
        if os.path.exists(lab) and os.path.getsize(lab):
            pairs.append((img, lab, 5))                      # new plates count 5x
    random.shuffle(pairs)
    n_val = len(pairs) // 8
    for split, items in (("val", pairs[:n_val]), ("train", pairs[n_val:])):
        os.makedirs(os.path.join(WORK, "images", split)); os.makedirs(os.path.join(WORK, "labels", split))
        for i, (img, lab, times) in enumerate(items):
            for k in range(times if split == "train" else 1):
                stem = f"{i:05d}_{k}"
                shutil.copy(img, os.path.join(WORK, "images", split, stem + os.path.splitext(img)[1]))
                shutil.copy(lab, os.path.join(WORK, "labels", split, stem + ".txt"))
    with open(os.path.join(WORK, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {WORK}\ntrain: images/train\nval: images/val\nnames:\n")
        f.writelines(f"  {i}: '{n}'\n" for i, n in enumerate(NAMES))
    print(f"{len(pairs)} plates ({sum(1 for p in pairs if p[2] == 5)} new)")


def main():
    from ultralytics import YOLO
    get_ealpr()
    build()
    current = os.path.join(ROOT, "models", "plate_chars.pt")
    model = YOLO(current)
    device = 0 if __import__("torch").cuda.is_available() else "cpu"
    model.train(data=os.path.join(WORK, "data.yaml"), imgsz=224, epochs=40, batch=64, device=device,
                fliplr=0.0, mosaic=0.0, degrees=4, translate=0.05, scale=0.25, hsv_v=0.4, patience=10,
                project=os.path.join(HERE, "runs"), name="plate_chars", exist_ok=True)
    best = os.path.join(HERE, "runs", "plate_chars", "weights", "best.pt")
    new_map = YOLO(best).val(data=os.path.join(WORK, "data.yaml"), imgsz=224, device=device).box.map50
    print(f"new model mAP50: {new_map:.3f}")
    if new_map >= 0.95:
        shutil.copy(current, os.path.join(ROOT, "models", "plate_chars_old.pt"))
        shutil.copy(best, current)
        print("Installed. The old model was kept as models/plate_chars_old.pt")
    else:
        print("Not installed: accuracy below 95%. Label more plates and try again.")


if __name__ == "__main__":
    sys.exit(main())
