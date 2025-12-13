import cv2
import numpy as np

def smart_map_cleanup(
    pgm_filepath,
    min_blob_size=50,
    connect_gap_size=3,
    prune_size=3,          # NEW: opening kernel to prune spikes
    prune_iters=1,         # NEW: increase to prune more
    unknown_val=205
):
    img = cv2.imread(pgm_filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading image")
        return

    # 1) walls mask (255 = wall)
    _, binary_map = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)

    # 2) keep only blobs above area threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    cleaned = np.zeros_like(binary_map)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_size:
            cleaned[labels == i] = 255

    # 3) close small gaps (your step)
    if connect_gap_size > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (connect_gap_size, connect_gap_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)

    # 4) NEW: prune tiny protrusions (opening shaves spikes)
    if prune_size > 0:
        k_prune = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (prune_size, prune_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_prune, iterations=prune_iters)

        # optional: re-close very lightly to keep borders connected
        if connect_gap_size > 0:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)

    # 5) reconstruct final map
    final_map = np.full_like(img, 255)

    # tighter unknown mask (avoid turning free space into unknown)
    unknown_mask = (img >= unknown_val - 2) & (img <= unknown_val + 2)
    final_map[unknown_mask] = unknown_val

    final_map[cleaned == 255] = 0

    out = pgm_filepath.replace(".pgm", "_smart_clean.pgm")
    cv2.imwrite(out, final_map)
    print("Saved:", out)

# Start gentle:
smart_map_cleanup(
    "final_map.pgm",
    min_blob_size=30,
    connect_gap_size=6,
    prune_size=3,
    prune_iters=1
)

# If spikes remain: set prune_iters=2 OR prune_size=5 (don’t jump too high).
