import os
import json
import csv
import torch
from datetime import datetime
from collections import Counter
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    LABEL_MAP,
    ID2LABEL,
    NUM_LABELS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    D1_MODEL_DIR,
    RESULTS_DIR_D1,
)

# =========================================================
# TEST PATH
# Written by new_train.py at the end of training.
# This is the only script that should ever read this file.
# =========================================================

TEST_PATH = os.path.join(RESULTS_DIR_D1, "test_split.json")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR_D1, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA GPU"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"✅ Device: {device_name}")

# =========================================================
# DATASET
# =========================================================

class InvoiceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words  = sample["words"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        encoding = self.tokenizer(
            words,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
            is_split_into_words=True,
        )

        word_ids       = encoding.word_ids(batch_index=0)
        aligned_bboxes = []
        aligned_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_labels.append(-100)
            else:
                aligned_bboxes.append(bboxes[word_idx])
                aligned_labels.append(labels[word_idx])

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "bbox":           torch.tensor(aligned_bboxes, dtype=torch.long),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }

# =========================================================
# EVALUATION — no confidence filtering
# Confidence filtering is a deployment decision only.
# See DEPLOY_CONFIDENCE_THRESHOLD in config.py.
# =========================================================

def run_evaluation(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                token_type_ids= batch["token_type_ids"].to(device),
                bbox=           batch["bbox"].to(device),
            )

            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, batch["labels"]):
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

    return all_preds, all_labels

# =========================================================
# CONFIDENCE DISTRIBUTION
# Use this to pick a deployment threshold — not a training
# one. See DEPLOY_CONFIDENCE_THRESHOLD in config.py.
# =========================================================

def confidence_distribution(model, loader):
    model.eval()
    all_confidences = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                token_type_ids= batch["token_type_ids"].to(device),
                bbox=           batch["bbox"].to(device),
            )

            probs      = torch.softmax(outputs.logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values

            for conf_seq, label_seq in zip(confidence, batch["labels"]):
                for c, l in zip(conf_seq.tolist(), label_seq.tolist()):
                    if l != -100:
                        all_confidences.append(c)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print(f"\n📊 Confidence distribution ({len(all_confidences)} predictions):")
    for t in thresholds:
        above = sum(1 for c in all_confidences if c >= t)
        pct   = 100 * above / len(all_confidences)
        print(f"   >= {t:.2f} : {above:6d} predictions ({pct:.1f}%)")

    print(f"\n   NOTE: Use DEPLOY_CONFIDENCE_THRESHOLD in config.py to set")
    print(f"   your deployment cutoff. Predictions below it should be")
    print(f"   flagged for human review, not silently dropped.")

# =========================================================
# SAVE REPORT
# =========================================================

def save_report(all_preds, all_labels):
    present_labels = sorted(set(all_labels))
    target_names   = [ID2LABEL[i] for i in present_labels]

    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    # ── Terminal output ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — TEST SET")
    print(f"{'='*60}")
    print(f"  Macro F1    : {macro_f1:.4f}  ← treats all classes equally")
    print(f"  Weighted F1 : {weighted_f1:.4f}  ← weighted by class frequency")
    print(f"\n  {'Label':<20} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}")
    print(f"  {'-'*55}")
    for label_name in target_names:
        m = report[label_name]
        print(f"  {label_name:<20} "
              f"{m['precision']:>7.3f} "
              f"{m['recall']:>7.3f} "
              f"{m['f1-score']:>7.3f} "
              f"{int(m['support']):>9}")

    # ── Text file ─────────────────────────────────────────
    txt_path = os.path.join(RESULTS_DIR_D1, f"eval_test_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(f"LAYOUTLM — FINAL TEST SET EVALUATION\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp   : {timestamp}\n")
        f.write(f"Model       : {D1_MODEL_DIR}\n")
        f.write(f"Device      : {device_name}\n\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n")
        f.write(f"Weighted F1 : {weighted_f1:.4f}\n\n")
        f.write(f"{'Label':<20} {'Precision':>10} {'Recall':>8} "
                f"{'F1':>8} {'Support':>10}\n")
        f.write("-" * 60 + "\n")
        for label_name in target_names:
            m = report[label_name]
            f.write(f"{label_name:<20} "
                    f"{m['precision']:>10.4f} "
                    f"{m['recall']:>8.4f} "
                    f"{m['f1-score']:>8.4f} "
                    f"{int(m['support']):>10}\n")
    print(f"\n  📄 Report saved      : {txt_path}")

    # ── Confusion matrix ───────────────────────────────────
    cm      = confusion_matrix(all_labels, all_preds, labels=present_labels)
    cm_path = os.path.join(RESULTS_DIR_D1, f"confusion_matrix_test_{timestamp}.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted"] + target_names)
        for name, row in zip(target_names, cm):
            writer.writerow([name] + list(row))
    print(f"  📊 Confusion matrix  : {cm_path}")

    # ── Top misclassifications ─────────────────────────────
    errors = [
        (ID2LABEL[t], ID2LABEL[p])
        for t, p in zip(all_labels, all_preds)
        if t != p
    ]
    error_counts = Counter(errors).most_common(10)

    print(f"\n  🔬 Top misclassifications (true → predicted):")
    for (true_label, pred_label), count in error_counts:
        print(f"     {true_label:<20} → {pred_label:<20} {count:5d} times")

    return macro_f1, weighted_f1

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"\n{'='*60}")
    print(f"  LAYOUTLM — FINAL TEST SET EVALUATION")
    print(f"  This is the honest number. Run this only once.")
    print(f"{'='*60}\n")

    # ── Load test set ──────────────────────────────────────
    print(f"📂 Loading test set: {TEST_PATH}")
    if not os.path.exists(TEST_PATH):
        print(f"\n❌ Test set not found: {TEST_PATH}")
        print(f"   Run new_train.py first — it saves the test split automatically.")
        return

    with open(TEST_PATH, "r") as f:
        test_data = json.load(f)
    print(f"   Samples: {len(test_data)}")

    # ── Load model ─────────────────────────────────────────
    print(f"\n🤖 Loading model: {D1_MODEL_DIR}")
    if not os.path.exists(D1_MODEL_DIR):
        print(f"\n❌ Model not found: {D1_MODEL_DIR}")
        print(f"   Run new_train.py first.")
        return

    tokenizer = LayoutLMTokenizerFast.from_pretrained(D1_MODEL_DIR)
    model     = LayoutLMForTokenClassification.from_pretrained(
        D1_MODEL_DIR, num_labels=NUM_LABELS
    ).to(device)

    test_loader = DataLoader(
        InvoiceDataset(test_data, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ── Evaluate ───────────────────────────────────────────
    all_preds, all_labels = run_evaluation(model, test_loader)

    # ── Save report ────────────────────────────────────────
    macro_f1, weighted_f1 = save_report(all_preds, all_labels)

    # ── Confidence distribution ────────────────────────────
    confidence_distribution(model, test_loader)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Results     : {RESULTS_DIR_D1}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()