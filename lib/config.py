from pathlib import Path
from pathlib import Path

HOME = Path.home()
RAW_DIR = HOME / 'datasets/eating_raw/'
NURSING_RAW_DIR = HOME / "datasets/nursingv1"
NURSING_LABEL_DIR = HOME / "datasets/eating_labels"
DEVICE = 'cuda:0'