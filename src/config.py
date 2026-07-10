from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PDF_PATH = DATA_DIR / "sample.pdf"