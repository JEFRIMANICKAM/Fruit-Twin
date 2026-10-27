import os
from typing import Dict, Any
from data.parse_real_dataset import get_real_dataset_summary, load_real_dataset

def get_dataset_summary() -> Dict[str, Any]:
    """
    Returns summary metrics of the real Nam Dok Mai Si Tong dataset at D:\\IOT LAB\\GAN FRUIT RIPENESS\\MANGO.
    """
    return get_real_dataset_summary()
