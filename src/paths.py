from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_LOCAL_DIR = PROJECT_ROOT / "output" / "local"
CONFIGS_DIR = PROJECT_ROOT / "configs"
PADDLEOCR_DIR = PROJECT_ROOT / "vendor" / "PaddleOCR"
