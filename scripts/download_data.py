"""Download MIT-BIH to data/raw/."""
from pathlib import Path

from src.data.load import download_mitbih

if __name__ == "__main__":
    download_mitbih(Path("data/raw/mitdb"))
    print("Done.")