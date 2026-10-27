import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

REAL_DATASET_DIR = r"D:\IOT LAB\GAN FRUIT RIPENESS\MANGO"
TRAIN_CSV = os.path.join(REAL_DATASET_DIR, "Train_Data.csv")
VALIDATE_CSV = os.path.join(REAL_DATASET_DIR, "Validate_Data.csv")

def assign_ripeness_stage(tss_ta: float, day: int, dafs: int) -> str:
    """
    Scientific mapping of Nam Dok Mai Si Tong mango biochemical parameters to 4 ripeness stages.
    """
    if pd.isna(tss_ta):
        # Infer from day + DAFS if TSS/TA missing (like in Validate set)
        if dafs <= 80 and day <= 3:
            return "Unripe"
        elif dafs <= 90 and day <= 5:
            return "Semi-Ripe"
        elif day >= 7 or dafs >= 100:
            return "Ripe" if day <= 7 else "Overripe"
        return "Semi-Ripe"

    if tss_ta < 2.5:
        return "Unripe"
    elif tss_ta < 4.0:
        return "Semi-Ripe"
    elif tss_ta < 7.5:
        return "Ripe"
    else:
        return "Overripe"

def load_real_dataset() -> Dict[str, pd.DataFrame]:
    """
    Loads train and validation data from D:\\IOT LAB\\GAN FRUIT RIPENESS\\MANGO.
    Appends derived ripeness_stage and estimated days to ripe.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Train_Data.csv not found at {TRAIN_CSV}")
        
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VALIDATE_CSV)
    
    # Process Train Data
    train_df["ripeness_stage"] = train_df.apply(
        lambda r: assign_ripeness_stage(r.get("TSS/TA"), r.get("Storage Time"), r.get("DAFS")), axis=1
    )
    # Estimate days to ripe (Day 7 is typically peak ripe in Nam Dok Mai Si Tong)
    train_df["days_to_ripe"] = train_df["Storage Time"].apply(lambda d: max(0, 7 - d))
    
    # Process Validation Data
    val_df["ripeness_stage"] = val_df.apply(
        lambda r: assign_ripeness_stage(r.get("TSS/TA", np.nan), r.get("Storage Time"), r.get("DAF")), axis=1
    )
    val_df["days_to_ripe"] = val_df["Storage Time"].apply(lambda d: max(0, 7 - d))
    
    return {
        "train": train_df,
        "validate": val_df
    }

def get_real_dataset_summary() -> Dict[str, Any]:
    """
    Returns rich dataset metrics from the real Kasetsart University dataset.
    """
    try:
        datasets = load_real_dataset()
        train_df = datasets["train"]
        val_df = datasets["validate"]
        
        stage_counts = train_df["ripeness_stage"].value_counts().to_dict()
        dafs_counts = train_df["DAFS"].value_counts().to_dict()
        
        return {
            "dataset_name": "Nam Dok Mai Si Tong Mango Dataset",
            "source": "Kasetsart University, Bangkok (Commercial orchard in Nakorn Ratchasima)",
            "total_samples": len(train_df) + len(val_df),
            "train_samples": len(train_df),
            "validate_samples": len(val_df),
            "parameters": ["Storage Time", "DAFS", "Weight (W)", "Voltage (V)", "Capacitance (C)", "R", "G", "B", "TA", "TSS", "TSS/TA"],
            "stage_distribution": stage_counts,
            "dafs_distribution": dafs_counts,
            "weight_stats": {
                "min": float(train_df["W"].min()),
                "max": float(train_df["W"].max()),
                "mean": round(float(train_df["W"].mean()), 1)
            },
            "tss_stats": {
                "min": float(train_df["TSS"].min()),
                "max": float(train_df["TSS"].max()),
                "mean": round(float(train_df["TSS"].mean()), 2)
            },
            "sample_train_records": train_df.head(15).to_dict(orient="records")
        }
    except Exception as e:
        return {
            "error": str(e),
            "dataset_name": "Nam Dok Mai Si Tong Mango Dataset",
            "total_samples": 0
        }

if __name__ == "__main__":
    res = get_real_dataset_summary()
    print("Dataset Summary:")
    print(f"Total Samples: {res['total_samples']}")
    print(f"Stages: {res['stage_distribution']}")
