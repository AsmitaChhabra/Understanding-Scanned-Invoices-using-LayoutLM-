import os
import json
import pandas as pd
import cv2
import pytesseract
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_DIR    = "Invoice_dataset_D1/D1_raw/batch_1/batch1_1"
CSV_PATH   = "Invoice_dataset_D1/D1_raw/batch_1/batch1_1.csv"
OUTPUT_DIR = "Invoice_dataset_D1/annotations/batch1_1"
TEST_MODE  = False
TEST_LIMIT = 5

os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

# ─── FIELD COLORS ─────────────────────────────────────────────────────────────
FIELD_COLORS = {
    "invoice_number":  (255, 0,   0  ),   # Red
    "invoice_date":    (0,   200, 0  ),   # Green
    "due_date":        (255, 255, 0  ),   # Yellow
    "client_name":     (0,   0,   255),   # Blue
    "client_address":  (255, 165, 0  ),   # Orange
    "client_tax_id":   (0,   200, 200),   # Teal
    "seller_name":     (128, 0,   128),   # Purple
    "seller_address":  (0,   255, 255),   # Cyan
    "seller_tax_id":   (200, 100, 0  ),   # Brown
    "tax":             (0,   128, 0  ),   # Dark Green
    "total":           (255, 20,  147),   # Pink
}

# ─── PARSE JSON ───────────────────────────────────────────────────────────────
def parse_json_cell(cell):
    try:
        cell = cell.strip()
        return json.loads(cell)
    except Exception as e:
        print(f"    JSON parse error: {e}")
        return None

# ─── FIND SINGLE LINE TEXT BBOX ───────────────────────────────────────────────
def find_text_bbox(ocr_data, search_text):
    if not search_text or str(search_text).strip() == "":
        return None

    search_text_normalized = str(search_text).strip().replace("$", "").replace(".", ",").strip()
    search_tokens = search_text_normalized.lower().split()

    if not search_tokens:
        return None

    words   = ocr_data['text']
    confs   = ocr_data['conf']
    lefts   = ocr_data['left']
    tops    = ocr_data['top']
    widths  = ocr_data['width']
    heights = ocr_data['height']
    n       = len(words)

    for i in range(n):
        word = str(words[i]).strip()
        if not word or int(confs[i]) < 40:
            continue

        word_normalized = word.lower().replace("$", "").replace(".", ",").strip()

        if word_normalized != search_tokens[0]:
            continue

        match_boxes = [(lefts[i], tops[i], widths[i], heights[i])]
        matched = 1
        j = i + 1

        while matched < len(search_tokens) and j < n:
            next_word = str(words[j]).strip().lower().replace("$", "").replace(".", ",")
            if next_word and next_word == search_tokens[matched]:
                match_boxes.append((lefts[j], tops[j], widths[j], heights[j]))
                matched += 1
            j += 1

        if matched >= max(1, len(search_tokens) // 2):
            x_min = min(b[0] for b in match_boxes)
            y_min = min(b[1] for b in match_boxes)
            x_max = max(b[0] + b[2] for b in match_boxes)
            y_max = max(b[1] + b[3] for b in match_boxes)
            return [x_min, y_min, x_max, y_max]

    return None

# ─── FIND MULTI-LINE TEXT BBOX ────────────────────────────────────────────────
def find_multiline_bbox(ocr_data, full_text):
    if not full_text or str(full_text).strip() == "":
        return None

    lines = [l.strip() for l in str(full_text).split("\n") if l.strip()]
    if not lines:
        return None

    all_boxes = []
    for line in lines:
        bbox = find_text_bbox(ocr_data, line)
        if bbox:
            all_boxes.append(bbox)

    if not all_boxes:
        return None

    x_min = min(b[0] for b in all_boxes)
    y_min = min(b[1] for b in all_boxes)
    x_max = max(b[2] for b in all_boxes)
    y_max = max(b[3] for b in all_boxes)

    return [x_min, y_min, x_max, y_max]

# ─── EXTRACT TAX ID FROM STRING ───────────────────────────────────────────────
def extract_tax_id(tax_string):
    """Extract just the ID number from strings like '945-82-2137'"""
    if not tax_string:
        return ""
    # Tax ID is usually after a colon e.g. "Tax Id: 945-82-2137"
    if ":" in str(tax_string):
        return tax_string.split(":")[-1].strip()
    return str(tax_string).strip()

# ─── ANNOTATE ONE IMAGE ───────────────────────────────────────────────────────
def annotate_image(img_path, json_data):
    image = cv2.imread(img_path)
    if image is None:
        print(f"    Could not read: {img_path}")
        return None, []

    rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

    # ── Correctly pull from nested "invoice" key ──
    invoice = json_data.get("invoice", {})

    # ── Pull tax and total from subtotal ──
    total_val = ""
    tax_val   = ""
    subtotal  = json_data.get("subtotal", {})
    if subtotal:
        total_val = str(subtotal.get("total", ""))
        tax_val   = str(subtotal.get("tax", ""))

    # ── Fallback: calculate total from items ──
    if not total_val:
        items = json_data.get("items", [])
        if items:
            try:
                total_val = str(round(sum(float(item.get("total_price", 0)) for item in items), 2))
            except:
                pass

    fields = {
        "invoice_number": (invoice.get("invoice_number", ""),  "single"),
        "invoice_date":   (invoice.get("invoice_date", ""),    "single"),
        "due_date":       (invoice.get("due_date", ""),        "single"),
        "client_name":    (invoice.get("client_name", ""),     "single"),
        "client_address": (invoice.get("client_address", ""),  "multi"),
        "seller_name":    (invoice.get("seller_name", ""),     "single"),
        "seller_address": (invoice.get("seller_address", ""),  "multi"),
        "tax":            (tax_val,                            "single"),
        "total":          (total_val,                          "single"),
    }

    print(f"    Fields: { {k:v[0] for k,v in fields.items() if v[0]} }")

    annotations = []

    for field_name, (field_value, field_type) in fields.items():
        if not field_value:
            continue

        if field_type == "multi":
            bbox = find_multiline_bbox(ocr_data, field_value)
        else:
            bbox = find_text_bbox(ocr_data, field_value)

        if bbox:
            annotations.append({
                "label": field_name,
                "value": str(field_value),
                "bbox":  bbox
            })
            color = FIELD_COLORS.get(field_name, (0, 255, 255))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, field_name, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print(f"    ⚠️  Could not find '{field_name}': {field_value}")

    return image, annotations

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n📂 Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, header=0)

    rows = df.head(TEST_LIMIT) if TEST_MODE else df
    print(f"📸 Processing {len(rows)} images (TEST_MODE={TEST_MODE})\n")

    all_annotations = []

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Annotating"):
        file_name = str(row.iloc[0]).strip()
        json_cell = str(row.iloc[1]).strip()

        if not file_name or file_name == "nan":
            continue

        json_data = parse_json_cell(json_cell)
        if not json_data:
            print(f"  ⚠️  Skipping {file_name} — bad JSON")
            continue

        img_path = os.path.join(IMG_DIR, file_name)
        if not os.path.exists(img_path):
            print(f"  ⚠️  Image not found: {img_path}")
            continue

        annotated_img, annotations = annotate_image(img_path, json_data)

        if annotated_img is not None:
            vis_path = os.path.join(OUTPUT_DIR, "visualizations", file_name)
            cv2.imwrite(vis_path, annotated_img)

            all_annotations.append({
                "file_name":   file_name,
                "annotations": annotations
            })

    out_json = os.path.join(OUTPUT_DIR, "annotations.json")
    with open(out_json, "w") as f:
        json.dump(all_annotations, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Annotations saved : {out_json}")
    print(f"   Visualizations    : {OUTPUT_DIR}/visualizations/")
    print(f"   Total processed   : {len(all_annotations)} invoices")

if __name__ == "__main__":
    main()