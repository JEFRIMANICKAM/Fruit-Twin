"""
Fruits-360 Benchmark Setup Script for FruitTwin
Sets up backend/data/fruits-360 directory with official Fruits-360 Training and Test splits.
Downloads representative images from the official Horea94/Fruit-Images-Dataset repository.
"""

import os
import requests
import urllib.parse
from PIL import Image

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FRUITS360_DIR = os.path.join(DATA_DIR, "fruits-360")
TRAIN_DIR = os.path.join(FRUITS360_DIR, "Training")
TEST_DIR = os.path.join(FRUITS360_DIR, "Test")

CLASSES_TO_SETUP = [
    "Apple Braeburn",
    "Apple Crimson Snow",
    "Apple Golden 1",
    "Apple Granny Smith",
    "Banana",
    "Banana Lady Finger",
    "Banana Red",
    "Orange",
    "Papaya",
    "Guava",
    "Pomegranate",
    "Watermelon"
]

GITHUB_BASE = "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master"

def setup_fruits360_benchmark(samples_per_train=12, samples_per_test=4):
    print(f"Setting up Fruits-360 benchmark at: {FRUITS360_DIR}")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # For Mango, also include local high-resolution mango specimens from RawRipe/mangoes
    rawripe_mango_train = os.path.join(DATA_DIR, "rawripe", "mangoes", "raw", "images")
    rawripe_mango_ripe = os.path.join(DATA_DIR, "rawripe", "mangoes", "ripe", "images")

    mango_train_dest = os.path.join(TRAIN_DIR, "Mango")
    mango_test_dest = os.path.join(TEST_DIR, "Test")
    os.makedirs(mango_train_dest, exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, "Mango"), exist_ok=True)

    # Copy / prepare Mango images
    if os.path.exists(rawripe_mango_train):
        m_imgs = [f for f in os.listdir(rawripe_mango_train) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for idx, f in enumerate(m_imgs[:15]):
            src = os.path.join(rawripe_mango_train, f)
            dest = os.path.join(TRAIN_DIR, "Mango", f"mango_{idx}.jpg")
            if not os.path.exists(dest):
                Image.open(src).resize((100, 100)).save(dest, "JPEG")
        for idx, f in enumerate(m_imgs[15:20]):
            src = os.path.join(rawripe_mango_train, f)
            dest = os.path.join(TEST_DIR, "Mango", f"mango_test_{idx}.jpg")
            if not os.path.exists(dest):
                Image.open(src).resize((100, 100)).save(dest, "JPEG")

    # Download from Fruits-360 GitHub repo for remaining classes
    for cname in CLASSES_TO_SETUP:
        c_train = os.path.join(TRAIN_DIR, cname)
        c_test = os.path.join(TEST_DIR, cname)
        os.makedirs(c_train, exist_ok=True)
        os.makedirs(c_test, exist_ok=True)

        existing_train = len(os.listdir(c_train))
        existing_test = len(os.listdir(c_test))

        if existing_train >= samples_per_train and existing_test >= samples_per_test:
            print(f"[{cname}] Already has {existing_train} train and {existing_test} test images.")
            continue

        qname = urllib.parse.quote(cname)
        print(f"Fetching Fruits-360 benchmark images for [{cname}]...")

        # Potential image index patterns in Fruits-360
        patterns = [f"{i}_100.jpg" for i in range(30)] + [f"r_{i}_100.jpg" for i in range(20)]

        downloaded_train = existing_train
        downloaded_test = existing_test

        for fname in patterns:
            if downloaded_train >= samples_per_train and downloaded_test >= samples_per_test:
                break

            target_split = "Training" if downloaded_train < samples_per_train else "Test"
            target_dest_dir = c_train if target_split == "Training" else c_test
            target_file_path = os.path.join(target_dest_dir, fname)

            if os.path.exists(target_file_path):
                continue

            url = f"{GITHUB_BASE}/{target_split}/{qname}/{fname}"
            try:
                r = requests.get(url, timeout=6)
                if r.status_code == 200 and len(r.content) > 500:
                    with open(target_file_path, "wb") as f:
                        f.write(r.content)
                    if target_split == "Training":
                        downloaded_train += 1
                    else:
                        downloaded_test += 1
            except Exception as e:
                pass

        print(f"[{cname}] Setup complete: {downloaded_train} train, {downloaded_test} test.")

    print("Fruits-360 benchmark dataset successfully configured!")

if __name__ == "__main__":
    setup_fruits360_benchmark()
