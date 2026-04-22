import json

INPUT_PATH = "Into_the_wild_D2/dataset_wild_unmapped.json"
OUTPUT_PATH = "Into_the_wild_D2//dataset_mapped.json"

# mapping from OLD LABEL ID → OLD LABEL NAME
OLD_ID2LABEL = {
    0: "O",
    1: "B-vendor",
    2: "B-address",
    3: "B-item",
    4: "B-item_price",
    5: "B-item_quant",
    6: "B-date",
    7: "B-total",
    8: "B-cgst",
    9: "B-sgst",
    10: "B-gstin",
}

# mapping from wild → clean label names
WILD_TO_CLEAN = {
    "B-vendor": "seller_name",
    "B-address": "seller_address",
    "B-date": "invoice_date",
    "B-total": "total",
    "B-cgst": "tax",
    "B-sgst": "tax",
    "B-gstin": "tax",
}

# final label map (clean)
LABEL_MAP = {
    "other": 0,
    "invoice_number": 1,
    "invoice_date": 2,
    "due_date": 3,
    "client_name": 4,
    "client_address": 5,
    "seller_name": 6,
    "seller_address": 7,
    "tax": 8,
    "total": 9,
}

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

for sample in data:
    new_labels = []
    
    for old_id in sample["labels"]:
        old_label = OLD_ID2LABEL.get(old_id, "O")
        
        clean_label_name = WILD_TO_CLEAN.get(old_label, "other")
        clean_label_id = LABEL_MAP[clean_label_name]
        
        new_labels.append(clean_label_id)
    
    sample["labels"] = new_labels
import os

os.makedirs("preprocessed/wild", exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)


print("✅ Label mapping complete!")