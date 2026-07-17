"""
data_prep.py
------------
Converts the raw Olivetti/AT&T Faces array into a folder-per-student image
layout (the standard format expected by face-recognition training pipelines,
and the same layout used by the reference repos in the project brief) and
generates a student roster (data/students.csv).

Dataset note
------------
The project brief points to two Kaggle datasets (CelebA, Faces-in-the-Wild).
Kaggle isn't reachable from this build environment. In its place we use the
AT&T/Olivetti Faces Dataset (40 subjects x 10 images, varying lighting,
expression, and head angle per subject) which is mirrored on GitHub and was
downloaded directly (`data/olivetti_faces.npy`, `data/olivetti_faces_target.npy`).
It's a smaller dataset but has the exact property this project needs: multiple
images per identity captured under different conditions, which is what step 9
of the brief ("test under different lighting and angles") requires.
Swapping in real student photos later only requires replacing the contents
of `data/raw_faces/<student_folder>/` — the rest of the pipeline is unchanged.
"""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"

random.seed(42)

# A plausible student roster standing in for the 40 anonymous Olivetti subjects.
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna",
    "Ishaan", "Rohan", "Ananya", "Diya", "Priya", "Isha", "Kavya", "Meera",
    "Sneha", "Riya", "Anika", "Tara", "Arnav", "Dev", "Kabir", "Yash",
    "Nikhil", "Rahul", "Karthik", "Varun", "Siddharth", "Aryan", "Pooja",
    "Neha", "Divya", "Sanya", "Aisha", "Lakshmi", "Gauri", "Nisha", "Swati", "Anjali",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Reddy", "Menon", "Gupta", "Rao",
    "Pillai", "Krishnan", "Pandey", "Das", "Joshi", "Mehta", "Kulkarni",
]


def generate_roster(n_students: int) -> list:
    used = set()
    roster = []
    for i in range(n_students):
        while True:
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            if name not in used:
                used.add(name)
                break
        roster.append({
            "student_id": f"STU{i+1:03d}",
            "name": name,
            "roll_no": f"CSE-{2027000 + i}",
            "class_label": i,  # maps to the Olivetti subject index
        })
    return roster


def export_images_and_roster():
    faces = np.load(DATA_DIR / "olivetti_faces.npy")   # (400, 64, 64) float32 in [0,1]
    targets = np.load(DATA_DIR / "olivetti_faces_target.npy")  # (400,) ints 0..39

    n_students = int(targets.max()) + 1
    roster = generate_roster(n_students)
    label_to_student = {r["class_label"]: r for r in roster}

    RAW_FACES_DIR.mkdir(exist_ok=True, parents=True)

    for idx, (img, label) in enumerate(zip(faces, targets)):
        student = label_to_student[int(label)]
        student_dir = RAW_FACES_DIR / student["student_id"]
        student_dir.mkdir(exist_ok=True)

        img_uint8 = (img * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8, mode="L")
        # image index within this subject's 10 photos (0..9)
        img_num = sum(1 for f in student_dir.glob("*.png"))
        im.save(student_dir / f"img_{img_num:02d}.png")

    with open(DATA_DIR / "students.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "name", "roll_no", "class_label"])
        writer.writeheader()
        writer.writerows(roster)

    print(f"Exported {len(faces)} images for {n_students} students to {RAW_FACES_DIR}")
    print(f"Roster written to {DATA_DIR / 'students.csv'}")
    return roster


if __name__ == "__main__":
    export_images_and_roster()
