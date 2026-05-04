import os
import json
import csv
import random
import torch
import numpy as np
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# CONFIG — change these when scaling to 3000 samples
# =========================================================

DATASET_PATH    = "Invoice_dataset_D1/preprocessed/batch1_1/dataset.json"
PRETRAINED_PATH = "microsoft/layoutlm-base-uncased"   # swap for your saved model path on re-runs
OUTPUT_DIR      = "models/layoutlm_D1_final"
RESULTS_DIR     = "results/D1"

# --- Data ---
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15   # locked — never used during training decisions
SEED            = 42

# --- Training ---
NUM_EPOCHS      = 15     # early stopping will decide actual stop point
BATCH_SIZE      = 2
MAX_SEQ_LEN     = 256
LEARNING_RATE   = 3e-5
WEIGHT_DECAY    = 0.01
PATIENCE        = 3      # stop if val F1 doesn't improve for 3 epochs
WARMUP_RATIO    = 0.1    # first 10% of steps = warmup
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.1

# --- Layer freezing ---
# Only train top N transformer layers + classifier
# Prevents overfitting on small dataset
UNFREEZE_LAYERS = [10, 11]   # layers 0-9 frozen, 10-11 + classifier trainable

# =========================================================
# LABEL MAP
# =========================================================

LABEL_MAP = {
    "other":          0,
    "invoice_number": 1,
    "invoice_date":   2,
    "due_date":       3,
    "client_name":    4,
    "client_address": 5,
    "seller_name":    6,
    "seller_address": 7,
    "tax":            8,
    "total":          9,
}

ID2LABEL   = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# =========================================================
# SETUP
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

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
# AUGMENTATION
# =========================================================

def add_noise_to_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x1 += random.randint(-5, 5)
    y1 += random.randint(-5, 5)
    x2 += random.randint(-5, 5)
    y2 += random.randint(-5, 5)
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def add_token_noise(word):
    if random.random() < 0.1:
        return word.replace("0", "O").replace("1", "I")
    return word

# =========================================================
# DATASET
# =========================================================

class InvoiceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=256, augment=False):
        self.samples     = samples
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.augment     = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.augment:
            words  = [add_token_noise(w) for w in sample["words"]]
            bboxes = [add_noise_to_bbox(b) for b in sample["bboxes"]]
        else:
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

        word_ids = encoding.word_ids(batch_index=0)

        aligned_bboxes = []
        aligned_labels = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens: [CLS], [SEP], padding
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_labels.append(-100)          # ignored by loss
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
# STRATIFIED SPLIT
# keeps rare classes in all three splits
# =========================================================

def stratified_split(data, seed=42):
    """
    Split by dominant label per sample so rare classes
    appear proportionally in train, val, and test.
    """
    def dominant_label(sample):
        non_other = [l for l in sample["labels"] if l != 0]
        if not non_other:
            return 0
        return Counter(non_other).most_common(1)[0][0]

    labels_for_split = [dominant_label(s) for s in data]

    # First split: train vs (val + test)
    train_data, temp_data, _, temp_labels = train_test_split(
        data, labels_for_split,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        random_state=seed,
        stratify=labels_for_split
    )

    # Second split: val vs test
    relative_test_size = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels
    )

    return train_data, val_data, test_data

# =========================================================
# TRUNCATION AUDIT
# tells you how much content is being cut off
# =========================================================

def audit_truncation(data, tokenizer, max_seq_len):
    truncated = 0
    for sample in data:
        encoding = tokenizer(
            sample["words"],
            is_split_into_words=True,
            add_special_tokens=True,
        )
        if len(encoding["input_ids"]) > max_seq_len:
            truncated += 1
    pct = 100 * truncated / len(data)
    print(f"   Truncation audit: {truncated}/{len(data)} samples exceed "
          f"{max_seq_len} tokens ({pct:.1f}%)")
    if pct > 20:
        print(f"   ⚠️  Over 20% truncated — consider increasing MAX_SEQ_LEN")

# =========================================================
# EVALUATION — no confidence filtering, both F1 metrics
# =========================================================

def evaluate(model, loader, split_name="Val"):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                token_type_ids= batch["token_type_ids"].to(device),
                bbox=           batch["bbox"].to(device),
            )

            # argmax — no confidence filtering
            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, batch["labels"]):
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:   # skip special tokens and padding
                        all_preds.append(p)
                        all_labels.append(l)

    if not all_labels:
        return 0.0, 0.0, {}

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

    return macro_f1, weighted_f1, report

# =========================================================
# CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
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

    present_labels = sorted(set(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    label_names = [ID2LABEL[i] for i in present_labels]

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + label_names)
        for name, row in zip(label_names, cm):
            writer.writerow([name] + list(row))

    print(f"   📊 Confusion matrix saved: {cm_path}")

# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(epoch_logs, final_report, best_macro_f1, best_weighted_f1, split_name="val"):
    # Training log CSV
    csv_path = os.path.join(RESULTS_DIR, f"training_log_{split_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "macro_f1", "weighted_f1", "saved"
        ])
        writer.writeheader()
        writer.writerows(epoch_logs)
    print(f"   📊 Training log: {csv_path}")

    # Summary text
    txt_path = os.path.join(RESULTS_DIR, f"summary_{split_name}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("LAYOUTLM D1 TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp   : {timestamp}\n")
        f.write(f"Device      : {device_name}\n")
        f.write(f"Dataset     : {DATASET_PATH}\n")
        f.write(f"Max Seq Len : {MAX_SEQ_LEN}\n")
        f.write(f"Epochs run  : {len(epoch_logs)}\n\n")
        f.write(f"Best Macro F1    : {best_macro_f1:.4f}\n")
        f.write(f"Best Weighted F1 : {best_weighted_f1:.4f}\n\n")
        f.write("Per-Class Metrics (best epoch)\n")
        f.write("-" * 40 + "\n")
        for label_name, metrics in final_report.items():
            if isinstance(metrics, dict) and label_name not in [
                "accuracy", "macro avg", "weighted avg"
            ]:
                f.write(
                    f"{label_name:<20} "
                    f"P: {metrics['precision']:.3f}  "
                    f"R: {metrics['recall']:.3f}  "
                    f"F1: {metrics['f1-score']:.3f}  "
                    f"Support: {int(metrics['support'])}\n"
                )
    print(f"   📄 Summary: {txt_path}")

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"\n{'='*55}")
    print(f"  LAYOUTLM D1 TRAINING")
    print(f"{'='*55}\n")

    # ── Load data ──────────────────────────────────────────
    print(f"📂 Loading: {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    print(f"   Total samples: {len(data)}")

    # ── Stratified split ───────────────────────────────────
    train_data, val_data, test_data = stratified_split(data, seed=SEED)
    print(f"   Train : {len(train_data)}")
    print(f"   Val   : {len(val_data)}")
    print(f"   Test  : {len(test_data)}  ← locked until final evaluation")

    # Save test split separately so it cannot be accidentally reused
    test_path = os.path.join(RESULTS_DIR, "test_split.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f)
    print(f"   Test split saved: {test_path}")

    # ── Load tokenizer ─────────────────────────────────────
    print(f"\n🤖 Loading tokenizer from: {PRETRAINED_PATH}")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(PRETRAINED_PATH)

    # ── Truncation audit ───────────────────────────────────
    print("\n🔍 Truncation audit:")
    audit_truncation(data, tokenizer, MAX_SEQ_LEN)

    # ── Model ──────────────────────────────────────────────
    print(f"\n🏗️  Loading model from: {PRETRAINED_PATH}")
    model = LayoutLMForTokenClassification.from_pretrained(
        PRETRAINED_PATH,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze top layers + classifier only
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        for layer_num in UNFREEZE_LAYERS:
            if f"encoder.layer.{layer_num}" in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    model.to(device)

    # ── Data loaders ───────────────────────────────────────
    train_loader = DataLoader(
        InvoiceDataset(train_data, tokenizer, MAX_SEQ_LEN, augment=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        InvoiceDataset(val_data, tokenizer, MAX_SEQ_LEN, augment=False),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Class weights ──────────────────────────────────────
    all_labels_flat = [l for s in train_data for l in s["labels"]]
    counts = Counter(all_labels_flat)
    total_tokens = sum(counts.values())

    weights = torch.tensor([
        np.log(1 + total_tokens / counts[i]) if counts[i] > 0 else 1.0
        for i in range(NUM_LABELS)
    ], dtype=torch.float32).to(device)

    print(f"\n📊 Label distribution (train):")
    for i in range(NUM_LABELS):
        pct = 100 * counts[i] / total_tokens if total_tokens > 0 else 0
        print(f"   {ID2LABEL[i]:<20} count: {counts[i]:6d}  ({pct:.1f}%)  "
              f"weight: {weights[i].item():.3f}")

    # ── Loss function ──────────────────────────────────────
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ── Optimizer + scheduler ──────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps   = len(train_loader) * NUM_EPOCHS
    warmup_steps  = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n⚙️  Training config:")
    print(f"   Epochs (max)  : {NUM_EPOCHS}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   LR            : {LEARNING_RATE}")
    print(f"   Weight decay  : {WEIGHT_DECAY}")
    print(f"   Warmup steps  : {warmup_steps}")
    print(f"   Total steps   : {total_steps}")
    print(f"   Patience      : {PATIENCE}")
    print(f"   Frozen layers : 0 - {min(UNFREEZE_LAYERS)-1}")

    # ── Training loop ──────────────────────────────────────
    best_macro_f1    = 0.0
    best_weighted_f1 = 0.0
    best_report      = {}
    patience_counter = 0
    epoch_logs       = []

    print(f"\n🚀 Starting training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            optimizer.zero_grad()   # zero before backward

            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                token_type_ids= batch["token_type_ids"].to(device),
                bbox=           batch["bbox"].to(device),
            )

            loss = loss_fct(
                outputs.logits.view(-1, NUM_LABELS),
                batch["labels"].to(device).view(-1),
            )

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate on both train and val to monitor overfitting
        train_macro, train_weighted, _ = evaluate(model, train_loader, "Train")
        val_macro,   val_weighted,   report = evaluate(model, val_loader, "Val")

        # Overfitting warning
        gap = train_macro - val_macro
        overfit_warning = "  ⚠️  Possible overfit" if gap > 0.10 else ""

        print(f"\nEpoch {epoch}")
        print(f"  Loss            : {avg_loss:.4f}")
        print(f"  Train Macro F1  : {train_macro:.4f}")
        print(f"  Val Macro F1    : {val_macro:.4f}  {overfit_warning}")
        print(f"  Val Weighted F1 : {val_weighted:.4f}")

        # Per-class F1 every epoch
        print(f"  Per-class F1:")
        for label_name, metrics in report.items():
            if isinstance(metrics, dict) and label_name not in [
                "accuracy", "macro avg", "weighted avg"
            ]:
                print(f"    {label_name:<20} F1: {metrics['f1-score']:.3f}  "
                      f"P: {metrics['precision']:.3f}  "
                      f"R: {metrics['recall']:.3f}")

        saved = False
        if val_macro > best_macro_f1:
            best_macro_f1    = val_macro
            best_weighted_f1 = val_weighted
            best_report      = report
            patience_counter = 0
            saved            = True
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  💾 Best model saved (Macro F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        epoch_logs.append({
            "epoch":       epoch,
            "train_loss":  round(avg_loss, 4),
            "macro_f1":    round(val_macro, 4),
            "weighted_f1": round(val_weighted, 4),
            "saved":       saved,
        })

    # ── Save training results ──────────────────────────────
    print(f"\n💾 Saving results...")
    save_results(epoch_logs, best_report, best_macro_f1, best_weighted_f1)

    # ── Final test set evaluation ──────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL TEST SET EVALUATION")
    print(f"  (This is the honest number — only run once)")
    print(f"{'='*55}\n")

    # Load best saved model for test evaluation
    best_model = LayoutLMForTokenClassification.from_pretrained(
        OUTPUT_DIR, num_labels=NUM_LABELS
    ).to(device)

    test_loader = DataLoader(
        InvoiceDataset(test_data, tokenizer, MAX_SEQ_LEN, augment=False),
        batch_size=BATCH_SIZE, shuffle=False
    )

    test_macro, test_weighted, test_report = evaluate(best_model, test_loader, "Test")

    print(f"  Test Macro F1    : {test_macro:.4f}")
    print(f"  Test Weighted F1 : {test_weighted:.4f}")
    print(f"\n  Per-class Test F1:")
    for label_name, metrics in test_report.items():
        if isinstance(metrics, dict) and label_name not in [
            "accuracy", "macro avg", "weighted avg"
        ]:
            print(f"    {label_name:<20} F1: {metrics['f1-score']:.3f}  "
                  f"Support: {int(metrics['support'])}")

    # Save test results
    save_results(epoch_logs, test_report, test_macro, test_weighted, split_name="test")
    save_confusion_matrix(best_model, test_loader)

    # ── Final summary ──────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*55}")
    print(f"  Best Val Macro F1  : {best_macro_f1:.4f}")
    print(f"  Test Macro F1      : {test_macro:.4f}")
    print(f"  Test Weighted F1   : {test_weighted:.4f}")
    print(f"  Model saved to     : {OUTPUT_DIR}")
    print(f"  Results saved to   : {RESULTS_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()