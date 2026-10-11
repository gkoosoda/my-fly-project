import numpy as np
from pathlib import Path

root = Path(__file__).parent   # <-- папка, где лежит test.py
ids = np.fromfile(root / "ids.bin", dtype="<u4")
groups = np.fromfile(root / "groups.bin", dtype="u1")
visible_ids = set(ids[groups < 3].tolist())
print(f"Видимых ID: {len(visible_ids)}")