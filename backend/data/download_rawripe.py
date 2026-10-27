import os
import sys
import time
import zipfile
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DATA_DIR, "RawRipe-main.zip")
DEST_DIR = os.path.join(DATA_DIR, "rawripe")
URL = "https://github.com/ai-agriculture-circuits-and-systems/RawRipe/archive/refs/heads/main.zip"

def download_and_extract():
    print(f"Target destination: {DEST_DIR}")
    if os.path.exists(DEST_DIR) and os.path.exists(os.path.join(DEST_DIR, "apples")):
        print(f"RawRipe dataset is already installed in {DEST_DIR}")
        return

    print(f"Downloading RawRipe from {URL}...")
    headers = {"User-Agent": "FruitTwin-Dataset-Downloader/1.0"}
    
    for attempt in range(1, 4):
        try:
            print(f"Download attempt {attempt}/3...")
            response = requests.get(URL, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            total_bytes = 0
            with open(ZIP_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes % (1024 * 1024 * 5) == 0:
                            print(f"Downloaded {total_bytes / (1024 * 1024):.1f} MB...")
            
            print(f"Download complete: {total_bytes / (1024 * 1024):.2f} MB saved to {ZIP_PATH}")
            break
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            time.sleep(3)

    print("Extracting zip archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    extracted_folder = os.path.join(DATA_DIR, "RawRipe-main")
    if os.path.exists(extracted_folder):
        if os.path.exists(DEST_DIR):
            import shutil
            shutil.rmtree(DEST_DIR, ignore_errors=True)
        os.rename(extracted_folder, DEST_DIR)
        print(f"Successfully unpacked RawRipe into {DEST_DIR}")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Cleaned up temporary zip file.")

if __name__ == "__main__":
    download_and_extract()
