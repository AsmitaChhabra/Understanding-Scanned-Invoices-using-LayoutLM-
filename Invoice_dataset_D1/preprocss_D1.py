import os
import json
import cv2
import pytesseract
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ANNOTATIONS_PATH = "/Users/asmita/Desktop/Understanding-Scanned-Invoices-using-LayoutLM/Invoice_dataset_D1/annotations/batch1_3/annotations_3.json"
IMG_DIR          = "Invoice_dataset_D1/D1_raw/batch1_3"
OUTPUT_DIR       = "Invoice_dataset_D1/preprocessed/batch1_3"
TARGET_SIZE      = (1000, 1400)  # standard size for all images (width, height)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LABEL MAP ────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "invoice_number": 1,
    "invoice_date":   2,
    "client_name":    3,
    "client_address": 4,
    "seller_name":    5,
    "seller_address": 6,
    "tax":            7,
    "total":          8,
    "other":          0,
}

# ─── NORMALIZE BBOX TO 0-1000 SCALE ──────────────────────────────────────────
def normalize_bbox(bbox, img_width, img_height):
    """LayoutLM expects bounding boxes normalized to 0-1000 scale"""
    x1, y1, x2, y2 = bbox
    return [
        int(1000 * x1 / img_width),
        int(1000 * y1 / img_height),
        int(1000 * x2 / img_width),
        int(1000 * y2 / img_height),
    ]

# ─── CHECK IF A WORD BBOX FALLS INSIDE AN ANNOTATION BBOX ────────────────────
def get_label_for_word(word_bbox, annotations, img_width, img_height):
    """Check if a word falls inside any annotation bounding box"""
    wx1 = word_bbox[0]
    wy1 = word_bbox[1]
    wx2 = word_bbox[0] + word_bbox[2]
    wy2 = word_bbox[1] + word_bbox[3]
    word_cx = (wx1 + wx2) / 2
    word_cy = (wy1 + wy2) / 2

    for ann in annotations:
        ax1, ay1, ax2, ay2 = ann["bbox"]
        if ax1 <= word_cx <= ax2 and ay1 <= word_cy <= ay2:
            return ann["label"]

    return "other"

# ─── PROCESS ONE IMAGE ────────────────────────────────────────────────────────
def process_image(img_path, annotations):
    # Load and resize image
    image = Image.open(img_path).convert("RGB")
    orig_width, orig_height = image.size
    image = image.resize(TARGET_SIZE)
    new_width, new_height = TARGET_SIZE

    # Scale factor for bboxes
    scale_x = new_width  / orig_width
    scale_y = new_height / orig_height

    # Scale annotation bboxes to new size
    scaled_annotations = []
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        scaled_annotations.append({
            "label": ann["label"],
            "bbox": [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ]
        })

    # Run OCR on resized image
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words       = []
    word_bboxes = []
    labels      = []

    for i in range(len(ocr_data['text'])):
        word = str(ocr_data['text'][i]).strip()
        conf = int(ocr_data['conf'][i])

        if not word or conf < 40:
            continue

        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        # Get label for this word
        label = get_label_for_word((x, y, w, h), scaled_annotations, new_width, new_height)

        # Normalize bbox to 0-1000 scale for LayoutLM
        norm_bbox = normalize_bbox([x, y, x+w, y+h], new_width, new_height)

        words.append(word)
        word_bboxes.append(norm_bbox)
        labels.append(LABEL_MAP.get(label, 0))

    return {
        "words":   words,
        "bboxes":  word_bboxes,
        "labels":  labels,
        "width":   new_width,
        "height":  new_height,
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n📂 Loading annotations: {ANNOTATIONS_PATH}")
    with open(ANNOTATIONS_PATH, "r") as f:
        all_annotations = json.load(f)

    print(f"📸 Processing {len(all_annotations)} images...\n")

    dataset = []

    for item in tqdm(all_annotations, desc="Preprocessing"):
        file_name   = item["file_name"]
        annotations = item["annotations"]
        img_path    = os.path.join(IMG_DIR, file_name)

        if not os.path.exists(img_path):
            print(f"  ⚠️  Image not found: {img_path}")
            continue

        processed = process_image(img_path, annotations)
        processed["file_name"] = file_name

        dataset.append(processed)

    # Save full preprocessed dataset
    out_path = os.path.join(OUTPUT_DIR, "dataset.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Dataset saved : {out_path}")
    print(f"   Total samples : {len(dataset)}")
    print(f"   Image size    : {TARGET_SIZE}")

    # Print a sample to verify
    if dataset:
        sample = dataset[0]
        print(f"\n📋 Sample entry (first image):")
        print(f"   File     : {sample['file_name']}")
        print(f"   Words    : {sample['words'][:5]} ...")
        print(f"   BBoxes   : {sample['bboxes'][:5]} ...")
        print(f"   Labels   : {sample['labels'][:5]} ...")

if __name__ == "__main__":
    main()