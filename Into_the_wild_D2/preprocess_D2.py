import json
import os
import sys

# Add project root to path so config.py is found
# regardless of which directory this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_D2 import LABEL_MAP, ID2LABEL

# =========================================================
# PATHS
# =========================================================

INPUT_PATH  = "Into_the_wild_D2/dataset_wild_unmapped.json"
OUTPUT_PATH = "Into_the_wild_D2/dataset_mapped.json"

# =========================================================
# D2 LABEL MAPPINGS
# These are D2-specific and stay here — not in config.
# config.py only holds the final clean label map.
# =========================================================

# D2 raw label ID → D2 label name
OLD_ID2LABEL = {
    0:  "O",
    1:  "B-vendor",
    2:  "B-address",
    3:  "B-item",
    4:  "B-item_price",
    5:  "B-item_quant",
    6:  "B-date",
    7:  "B-total",
    8:  "B-cgst",
    9:  "B-sgst",
    10: "B-gstin",
}

# D2 label name → clean label name (config vocabulary)
# Labels not listed here map to "other"
# cgst + sgst + gstin all collapse into "tax" — D1 has one tax class
WILD_TO_CLEAN = {
    "B-vendor":  "seller_name",
    "B-address": "seller_address",
    "B-date":    "invoice_date",
    "B-total":   "total",
    "B-cgst":    "tax",
    "B-sgst":    "tax",
    "B-gstin":   "tax",
}

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"\n📂 Loading: {INPUT_PATH}")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    print(f"   Samples: {len(data)}")

    for sample in data:
        new_labels = []
        for old_id in sample["labels"]:
            old_label_name  = OLD_ID2LABEL.get(old_id, "O")
            clean_label_name = WILD_TO_CLEAN.get(old_label_name, "other")
            clean_label_id   = LABEL_MAP[clean_label_name]
            new_labels.append(clean_label_id)
        sample["labels"] = new_labels

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # ── Verify label distribution ──────────────────────────
    from collections import Counter

    all_labels  = [l for s in data for l in s["labels"]]
    counts      = Counter(all_labels)
    total_tokens = sum(counts.values())

    print(f"\n   Label distribution after mapping:")
    for label_id in sorted(counts):
        name = ID2LABEL.get(label_id, f"id_{label_id}")
        pct  = 100 * counts[label_id] / total_tokens
        print(f"   {name:<20} count: {counts[label_id]:6d}  ({pct:.1f}%)")

    print(f"\n✅ Label mapping complete → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()