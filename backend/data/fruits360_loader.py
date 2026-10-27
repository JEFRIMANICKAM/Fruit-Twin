"""
Fruits-360 Dataset Loader & Dynamic Discovery Module for FruitTwin

Purpose:
- Automatically discover classes and varieties from any local Fruits-360 dataset directory
- Handle directory variations ('Training'/'Test', 'train'/'test', nested folders)
- Parse class folders into main fruit category and variety (e.g. 'Apple Braeburn' -> Apple, Braeburn)
- Provide train and test splits for model training and evaluation
- Provide a benchmark sample loader / downloader for automated testing
"""

import os
import glob
import re
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

DEFAULT_DATASET_DIRS = [
    os.environ.get("FRUITS360_PATH"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fruits-360"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fruits-360_dataset", "fruits-360"),
    r"D:\IOT LAB\fruits-360",
    r"C:\fruits-360",
]

def find_fruits360_root() -> Optional[str]:
    """Finds the root directory containing Fruits-360 training and test data."""
    for path in DEFAULT_DATASET_DIRS:
        if path and os.path.isdir(path):
            # Check for Training/Test or train/test
            subdirs = [d.lower() for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if any(s in subdirs for s in ["training", "train"]) and any(s in subdirs for s in ["test", "validation"]):
                return os.path.abspath(path)
            # Check if nested
            for sub in os.listdir(path):
                nested = os.path.join(path, sub)
                if os.path.isdir(nested):
                    sub_subdirs = [d.lower() for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
                    if any(s in sub_subdirs for s in ["training", "train"]):
                        return os.path.abspath(nested)
    return None

def parse_class_name(class_folder_name: str) -> Tuple[str, Optional[str]]:
    """
    Parses Fruits-360 folder name into (fruit_category, variety_name).
    Examples:
      'Apple Braeburn' -> ('Apple', 'Braeburn')
      'Apple Granny Smith' -> ('Apple', 'Granny Smith')
      'Banana Red' -> ('Banana', 'Red (Sevazhai)')
      'Banana' -> ('Banana', 'Standard Cavendish')
      'Mango' -> ('Mango', 'Standard Mango')
      'Mango Red' -> ('Mango', 'Red Mango')
      'Orange' -> ('Orange', 'Standard Orange')
      'Pomegranate' -> ('Pomegranate', 'Standard Pomegranate')
      'Watermelon' -> ('Watermelon', 'Standard Watermelon')
    """
    clean_name = class_folder_name.strip()
    
    # Common fruit categories in Fruits-360
    known_categories = [
        "Apple", "Banana", "Mango", "Orange", "Papaya", "Guava", 
        "Pomegranate", "Watermelon", "Strawberry", "Coconut", "Litchi",
        "Pineapple", "Kiwi", "Avocado", "Lemon", "Lime", "Peach", 
        "Pear", "Plum", "Grape", "Cherry", "Fig", "Cantaloupe"
    ]
    
    for cat in known_categories:
        if clean_name.lower().startswith(cat.lower()):
            variety_part = clean_name[len(cat):].strip()
            # Clean up leading dash or underscore or space
            variety_part = re.sub(r"^[\s\-_]+", "", variety_part).strip()
            
            if not variety_part:
                variety_part = f"Standard {cat}"
            elif cat.lower() == "banana" and "red" in variety_part.lower():
                variety_part = "Red (Sevazhai)"
            elif cat.lower() == "banana" and "lady" in variety_part.lower():
                variety_part = "Lady Finger"
            
            return cat, variety_part
            
    # If no prefix matched, split first word as fruit and rest as variety
    parts = clean_name.split()
    if len(parts) > 1:
        return parts[0].capitalize(), " ".join(parts[1:])
    return clean_name.capitalize(), f"Standard {clean_name.capitalize()}"

def discover_dataset_classes(dataset_root: str) -> Dict[str, Any]:
    """
    Inspects actual directory structure and dynamically discovers available classes,
    varieties, splits, and file counts.
    """
    train_dir = None
    test_dir = None
    
    for item in os.listdir(dataset_root):
        full_p = os.path.join(dataset_root, item)
        if os.path.isdir(full_p):
            if item.lower() in ["training", "train"]:
                train_dir = full_p
            elif item.lower() in ["test", "validation"]:
                test_dir = full_p
                
    if not train_dir:
        train_dir = dataset_root
        
    classes_info = {}
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_names.sort()
    
    for idx, cname in enumerate(class_names):
        cat, variety = parse_class_name(cname)
        train_c_path = os.path.join(train_dir, cname)
        test_c_path = os.path.join(test_dir, cname) if test_dir and os.path.isdir(os.path.join(test_dir, cname)) else None
        
        train_count = len([f for f in os.listdir(train_c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(test_c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if test_c_path else 0
        
        classes_info[cname] = {
            "index": idx,
            "folder_name": cname,
            "fruit": cat,
            "variety": variety,
            "train_samples": train_count,
            "test_samples": test_count,
            "train_path": train_c_path,
            "test_path": test_c_path
        }
        
    return {
        "dataset_root": dataset_root,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "num_classes": len(classes_info),
        "classes": classes_info,
        "class_list": class_names
    }
