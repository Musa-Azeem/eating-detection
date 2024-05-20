import os
from pathlib import Path
import re

cwd = Path('/home/musa/datasets/eating_raw')
batch_date = '11-10-23'

files = [f for f in Path(cwd).iterdir() if f.is_file()]

for file in files:
    date_match = re.search(r'^.*-(\d\d-\d\d_\d\d_\d\d_\d\d)\..*$', file.name)
    if date_match:
        date = date_match.groups()[0]
        date_dir = cwd / Path(f'delta_batch-{batch_date}') / Path(date)
        if not os.path.exists(date):
            date_dir.mkdir()
        file.rename(date_dir / file.name)