"""
RawRipe Dataset Loader & Evaluation Split Reader for FruitTwin

Handles the official RawRipe dataset structure across all 10 fruit types:
- apples
- bananas
- coconuts
- guavas
- litchis
- mangoes
- oranges
- papayas
- pomegranates
- strawberries

Maturity States:
- raw
- ripe

Splits:
- Uses the official sets/train.txt and sets/test.txt files from the repository
"""

import os
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

DEFAULT_RAWRIPE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rawripe")

SUPPORTED_RAWRIPE_FRUITS = [
    "apples", "bananas", "coconuts", "guavas", "litchis",
    "mangoes", "oranges", "papayas", "pomegranates", "strawberries"
]

FRUIT_NAME_NORMALIZATION = {
    "apple": "apples", "apples": "apples",
    "banana": "bananas", "bananas": "bananas",
    "coconut": "coconuts", "coconuts": "coconuts",
    "guava": "guavas", "guavas": "guavas",
    "litchi": "litchis", "litchis": "litchis", "lychee": "litchis",
    "mango": "mangoes", "mangoes": "mangoes",
    "orange": "oranges", "oranges": "oranges",
    "papaya": "papayas", "papayas": "papayas",
    "pomegranate": "pomegranates", "pomegranates": "pomegranates",
    "strawberry": "strawberries", "strawberries": "strawberries"
}

def normalize_fruit_name(fruit_name: str) -> Optional[str]:
    """Normalizes any fruit name variant to the RawRipe folder name."""
    if not fruit_name:
        return None
    return FRUIT_NAME_NORMALIZATION.get(fruit_name.strip().lower())

def is_fruit_supported_by_rawripe(fruit_name: str) -> bool:
    """Checks whether the given fruit name has RawRipe maturity ground truth."""
    return normalize_fruit_name(fruit_name) is not None

def get_rawripe_root() -> Optional[str]:
    """Finds the root directory containing RawRipe dataset folders."""
    candidates = [
        os.environ.get("RAWRIPE_PATH"),
        DEFAULT_RAWRIPE_DIR,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "RawRipe-main"),
        r"D:\IOT LAB\RawRipe",
        r"C:\RawRipe",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            # Verify it contains at least one of the fruit folders
            if any(os.path.isdir(os.path.join(p, f)) for f in ["apples", "bananas", "mangoes"]):
                return os.path.abspath(p)
    return None

def load_rawripe_records(dataset_root: Optional[str] = None, fruit_filter: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads dataset records using the official sets/train.txt and sets/test.txt splits.
    Returns:
      {
        "train": [ { "fruit": "apples", "state": "raw", "image_path": "...", "label": 0 }, ... ],
        "test": [ ... ]
      }
    """
    root = dataset_root or get_rawripe_root()
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"RawRipe dataset not found at: {root}")

    train_records = []
    test_records = []
    
    target_fruits = [normalize_fruit_name(fruit_filter)] if fruit_filter else SUPPORTED_RAWRIPE_FRUITS
    target_fruits = [f for f in target_fruits if f]

    for fruit in target_fruits:
        fruit_dir = os.path.join(root, fruit)
        if not os.path.isdir(fruit_dir):
            continue

        for state in ["raw", "ripe"]:
            state_dir = os.path.join(fruit_dir, state)
            if not os.path.isdir(state_dir):
                continue

            img_dir = os.path.join(state_dir, "images")
            sets_dir = os.path.join(state_dir, "sets")

            # Check for train.txt and test.txt
            train_txt = os.path.join(sets_dir, "train.txt")
            test_txt = os.path.join(sets_dir, "test.txt")

            train_basenames = set()
            test_basenames = set()

            if os.path.exists(train_txt):
                with open(train_txt, "r", encoding="utf-8") as f:
                    train_basenames = {line.strip() for line in f if line.strip()}

            if os.path.exists(test_txt):
                with open(test_txt, "r", encoding="utf-8") as f:
                    test_basenames = {line.strip() for line in f if line.strip()}

            # If sets are missing, fallback to 80/20 directory split
            if os.path.isdir(img_dir):
                all_imgs = [img for img in os.listdir(img_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in all_imgs:
                    base = os.path.splitext(img_name)[0]
                    full_p = os.path.join(img_dir, img_name)
                    item = {
                        "fruit": fruit,
                        "state": state,  # "raw" or "ripe"
                        "label": 0 if state == "raw" else 1,
                        "image_path": full_p,
                        "image_name": img_name
                    }
                    if base in train_basenames:
                        train_records.append(item)
                    elif base in test_basenames:
                        test_records.append(item)
                    else:
                        # If basename not explicitly in txt files, partition deterministically
                        if hash(base) % 5 == 0:
                            test_records.append(item)
                        else:
                            train_records.append(item)

    return {
        "train": train_records,
        "test": test_records,
        "total_train": len(train_records),
        "total_test": len(test_records)
    }

def get_rawripe_summary(dataset_root: Optional[str] = None) -> Dict[str, Any]:
    """Generates comprehensive summary metrics for RawRipe dataset."""
    root = dataset_root or get_rawripe_root()
    if not root:
        return {"status": "not_installed", "message": "RawRipe dataset not yet downloaded."}

    summary = {
        "status": "ready",
        "dataset_name": "RawRipe Multi-Fruit Maturity Dataset",
        "root_directory": root,
        "supported_fruits": SUPPORTED_RAWRIPE_FRUITS,
        "fruits_stats": {}
    }

    try:
        records = load_rawripe_records(root)
        summary["total_train"] = records["total_train"]
        summary["total_test"] = records["total_test"]
        summary["total_images"] = records["total_train"] + records["total_test"]

        # Aggregate counts per fruit and state
        for rec in records["train"] + records["test"]:
            fr = rec["fruit"]
            st = rec["state"]
            if fr not in summary["fruits_stats"]:
                summary["fruits_stats"][fr] = {"raw": 0, "ripe": 0, "total": 0}
            summary["fruits_stats"][fr][st] += 1
            summary["fruits_stats"][fr]["total"] += 1

    except Exception as e:
        summary["error"] = str(e)

    return summary
