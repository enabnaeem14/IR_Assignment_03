import os
from dotenv import load_dotenv

load_dotenv()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise Exception(" Kaggle credentials missing in .env")

# Set Kaggle env variables
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

print("Kaggle Credentials Loaded!")

import subprocess

def download_dataset():
    os.makedirs("data", exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "asad1m9a9h6mood/news-articles",
        "-p", "data",
        "--unzip"
    ]
    print("Downloading dataset via Kaggle API...")
    subprocess.run(cmd)
    print("Dataset downloaded into /data")

if __name__ == "__main__":
    download_dataset()
