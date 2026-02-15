# 3. Class ID Mapping

import numpy as np

CLASS_IDS = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
NUM_CLASSES = len(CLASS_IDS)

# Build a lookup table for fast remapping (max value is 10000, so table size is 10001)
_REMAP_LUT = np.full(10001, 0, dtype=np.int64)  # default to 0 for any unexpected value
for contiguous_label, original_id in enumerate(CLASS_IDS):
    _REMAP_LUT[original_id] = contiguous_label


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Remap raw mask pixel values to contiguous class indices [0, NUM_CLASSES-1]."""
    return _REMAP_LUT[mask]


print(f"Classes: {NUM_CLASSES}")
print(f"ID mapping: {dict(zip(CLASS_IDS, range(NUM_CLASSES)))}")
