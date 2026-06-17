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

import glob

from config import (
    SEED,
    LABEL_MAP,
    ID2LABEL,
    NUM_LABELS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    VAL_SPLIT,
    TEST_SPLIT,
    D1_LEARNING_RATE,
    D1_NUM_EPOCHS,
    D1_PATIENCE,
    BASE_MODEL,
    D1_MODEL_DIR,
    RESULTS_DIR_D1,
    D1_PREPROCESSED_BASE,
)

# =========================================================
# TRAINING HYPERPARAMETERS
# (not in config because these are tuning knobs, not
#  shared constants — only new_train.py cares about them)
# =========================================================

WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
LABEL_SMOOTHING = 0.05        # reduced — minority classes need more confident signal
DROPOUT         = 0.1
UNFREEZE_LAYERS = [8, 9, 10, 11]   # more capacity for address/monetary fields

# =========================================================
# SETUP
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(D1_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_D1, exist_ok=True)

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
    def __init__(self, samples, tokenizer, max_seq_len=MAX_SEQ_LEN, augment=False):
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
# STRATIFIED SPLIT
# Ensures rare classes (tax, total) appear in all splits.
# Uses dominant non-other label per sample as the
# stratification key.
# =========================================================

def stratified_split(data, seed=SEED):
    def dominant_label(sample):
        non_other = [l for l in sample["labels"] if l != 0]
        if not non_other:
            return 0
        return Counter(non_other).most_common(1)[0][0]

    labels_for_split = [dominant_label(s) for s in data]

    # First cut: train vs (val + test)
    train_data, temp_data, _, temp_labels = train_test_split(
        data, labels_for_split,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        random_state=seed,
        stratify=labels_for_split,
    )

    # Second cut: val vs test
    relative_test_size   = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    temp_label_counts    = Counter(temp_labels)
    can_stratify         = all(c >= 2 for c in temp_label_counts.values())

    if can_stratify:
        val_data, test_data = train_test_split(
            temp_data,
            test_size=relative_test_size,
            random_state=seed,
            stratify=temp_labels,
        )
    else:
        print("   ⚠️  Some classes too rare for stratified val/test split — using random.")
        val_data, test_data = train_test_split(
            temp_data,
            test_size=relative_test_size,
            random_state=seed,
        )

    return train_data, val_data, test_data

# =========================================================
# TRUNCATION AUDIT
# =========================================================

def audit_truncation(data, tokenizer):
    truncated = 0
    for sample in data:
        encoding = tokenizer(
            sample["words"],
            is_split_into_words=True,
            add_special_tokens=True,
        )
        if len(encoding["input_ids"]) > MAX_SEQ_LEN:
            truncated += 1
    pct = 100 * truncated / len(data)
    print(f"   {truncated}/{len(data)} samples exceed {MAX_SEQ_LEN} tokens ({pct:.1f}%)")
    if pct > 20:
        print(f"   ⚠️  Over 20% truncated — consider increasing MAX_SEQ_LEN in config.py")

# =========================================================
# EVALUATE — no confidence filtering
# Confidence filtering is a deployment decision only.
# See DEPLOY_CONFIDENCE_THRESHOLD in config.py.
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

            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, batch["labels"]):
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:
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
    label_names    = [ID2LABEL[i] for i in present_labels]
    cm             = confusion_matrix(all_labels, all_preds, labels=present_labels)

    cm_path = os.path.join(RESULTS_DIR_D1, f"confusion_matrix_{timestamp}.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + label_names)
        for name, row in zip(label_names, cm):
            writer.writerow([name] + list(row))

    print(f"   📊 Confusion matrix saved: {cm_path}")

# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(epoch_logs, final_report, best_macro_f1, best_weighted_f1,
                 split_name="val"):

    csv_path = os.path.join(RESULTS_DIR_D1, f"training_log_{split_name}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "macro_f1", "weighted_f1", "saved"
        ])
        writer.writeheader()
        writer.writerows(epoch_logs)
    print(f"   📊 Training log : {csv_path}")

    txt_path = os.path.join(RESULTS_DIR_D1, f"summary_{split_name}_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("LAYOUTLM D1 TRAINING SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Timestamp        : {timestamp}\n")
        f.write(f"Device           : {device_name}\n")
        f.write(f"Dataset          : {D1_PREPROCESSED_BASE}\n")
        f.write(f"Max Seq Len      : {MAX_SEQ_LEN}\n")
        f.write(f"Epochs run       : {len(epoch_logs)}\n\n")
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
    print(f"   📄 Summary      : {txt_path}")

# =========================================================
# MAIN
# =========================================================

def main():
    print(f"\n{'='*55}")
    print(f"  LAYOUTLM D1 TRAINING")
    print(f"{'='*55}\n")

    # ── Load data — auto-discovers all batch dataset.json files ───
    dataset_files = sorted(glob.glob(
        os.path.join(D1_PREPROCESSED_BASE, "**", "preprocessed_dataset.json"),
        recursive=True
    ))

    if not dataset_files:
        print(f"❌ No dataset.json files found under {D1_PREPROCESSED_BASE}")
        return

    print(f"📂 Loading datasets from: {D1_PREPROCESSED_BASE}")
    data = []
    for path in dataset_files:
        with open(path, "r") as f:
            batch = json.load(f)
        print(f"   {path} — {len(batch)} samples")
        data += batch

    print(f"   Total samples: {len(data)}")

    # ── Remap old 10-class label IDs to new 9-class ────────
    # Old preprocss_D1.py used a 10-class map with due_date=3
    # which shifted all labels after it by 1.
    # This corrects any dataset preprocessed with the old map.
    OLD_TO_NEW = {
        0: 0,   # other          -> other
        1: 1,   # invoice_number -> invoice_number
        2: 2,   # invoice_date   -> invoice_date
        3: 0,   # due_date       -> other (dropped)
        4: 3,   # client_name    -> client_name
        5: 4,   # client_address -> client_address
        6: 5,   # seller_name    -> seller_name
        7: 6,   # seller_address -> seller_address
        8: 7,   # tax            -> tax
        9: 8,   # total          -> total
    }
    valid_ids   = set(LABEL_MAP.values())
    needs_remap = any(
        l not in valid_ids
        for s in data for l in s["labels"]
    )
    if needs_remap:
        print(f"   ⚠️  Old 10-class label IDs detected — remapping to 9-class...")
        for sample in data:
            sample["labels"] = [OLD_TO_NEW.get(l, 0) for l in sample["labels"]]
        print(f"   ✅ Remapping complete")
    else:
        print(f"   ✅ Label IDs already in 9-class format")

    # ── Stratified split ───────────────────────────────────
    print(f"\n📊 Splitting data (70 / 15 / 15)...")
    train_data, val_data, test_data = stratified_split(data, seed=SEED)
    print(f"   Train : {len(train_data)}")
    print(f"   Val   : {len(val_data)}")
    print(f"   Test  : {len(test_data)}  ← locked until evaluate.py")

    # ── Save test split — locked, do not reload during training ──
    test_path = os.path.join(RESULTS_DIR_D1, "test_split.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f)
    print(f"   Test split saved: {test_path}")

    # ── Tokenizer ──────────────────────────────────────────
    print(f"\n🤖 Loading tokenizer: {BASE_MODEL}")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_MODEL)

    # ── Truncation audit ───────────────────────────────────
    print(f"\n🔍 Truncation audit:")
    audit_truncation(data, tokenizer)

    # ── Model ──────────────────────────────────────────────
    print(f"\n🏗️  Loading model: {BASE_MODEL}")
    model = LayoutLMForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )

    # Freeze all layers first, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        for layer_num in UNFREEZE_LAYERS:
            if f"encoder.layer.{layer_num}" in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"   Frozen layers    : 0 – {min(UNFREEZE_LAYERS) - 1}")
    print(f"   Unfrozen layers  : {UNFREEZE_LAYERS} + classifier")

    model.to(device)

    # ── Data loaders ───────────────────────────────────────
    train_loader = DataLoader(
        InvoiceDataset(train_data, tokenizer, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        InvoiceDataset(val_data, tokenizer, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ── Minority class upsampling ──────────────────────────────
    # Samples rich in minority tokens (tax, total, addresses)
    # are duplicated to reduce other-dominance at sample level.
    MINORITY_LABELS = {
        LABEL_MAP["tax"], LABEL_MAP["total"],
        LABEL_MAP["client_address"], LABEL_MAP["seller_address"],
    }
    UPSAMPLE_FACTOR = 2

    upsampled = []
    for sample in train_data:
        minority_count = sum(1 for l in sample["labels"] if l in MINORITY_LABELS)
        total_count    = len(sample["labels"])
        minority_ratio = minority_count / total_count if total_count > 0 else 0
        if minority_ratio > 0.05:
            upsampled.extend([sample] * UPSAMPLE_FACTOR)
        else:
            upsampled.append(sample)

    print(f"\n📊 Upsampling:")
    print(f"   Original train samples : {len(train_data)}")
    print(f"   After upsampling       : {len(upsampled)}")
    train_data = upsampled

    # ── Class weights ──────────────────────────────────────
    all_labels_flat = [l for s in train_data for l in s["labels"]]
    counts          = Counter(all_labels_flat)
    total_tokens    = sum(counts.values())

    weights_list = []
    for i in range(NUM_LABELS):
        if counts[i] == 0:
            weights_list.append(1.0)
        else:
            w = np.log(1 + total_tokens / counts[i])
            if i in {LABEL_MAP["total"], LABEL_MAP["tax"]}:
                w *= 1.5   # extra boost for worst performers
            weights_list.append(w)

    weights = torch.tensor(weights_list, dtype=torch.float32).to(device)

    print(f"\n📊 Label distribution (train after upsampling):")
    for i in range(NUM_LABELS):
        pct = 100 * counts[i] / total_tokens if total_tokens > 0 else 0
        print(f"   {ID2LABEL[i]:<20} count: {counts[i]:6d}  ({pct:.1f}%)  "
              f"weight: {weights[i].item():.3f}")

    # ── Loss ───────────────────────────────────────────────
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ── Optimizer + scheduler ──────────────────────────────
    optimizer     = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=D1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps   = len(train_loader) * D1_NUM_EPOCHS
    warmup_steps  = int(WARMUP_RATIO * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n⚙️  Training config:")
    print(f"   Max epochs    : {D1_NUM_EPOCHS}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   Learning rate : {D1_LEARNING_RATE}")
    print(f"   Weight decay  : {WEIGHT_DECAY}")
    print(f"   Warmup steps  : {warmup_steps} / {total_steps}")
    print(f"   Patience      : {D1_PATIENCE}")

    # ── Training loop ──────────────────────────────────────
    best_macro_f1    = 0.0
    best_weighted_f1 = 0.0
    best_report      = {}
    patience_counter = 0
    epoch_logs       = []

    print(f"\n🚀 Starting training...\n")

    for epoch in range(1, D1_NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{D1_NUM_EPOCHS}"):
            optimizer.zero_grad()

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
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        train_macro, train_weighted, _      = evaluate(model, train_loader, "Train")
        val_macro,   val_weighted,   report = evaluate(model, val_loader,   "Val")

        gap              = train_macro - val_macro
        overfit_warning  = "  ⚠️  Possible overfit" if gap > 0.10 else ""

        print(f"\nEpoch {epoch}")
        print(f"  Loss            : {avg_loss:.4f}")
        print(f"  Train Macro F1  : {train_macro:.4f}")
        print(f"  Val Macro F1    : {val_macro:.4f}{overfit_warning}")
        print(f"  Val Weighted F1 : {val_weighted:.4f}")
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
            model.save_pretrained(D1_MODEL_DIR)
            tokenizer.save_pretrained(D1_MODEL_DIR)
            print(f"  💾 Best model saved (Macro F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{D1_PATIENCE}")
            if patience_counter >= D1_PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        epoch_logs.append({
            "epoch":       epoch,
            "train_loss":  round(avg_loss, 4),
            "macro_f1":    round(val_macro, 4),
            "weighted_f1": round(val_weighted, 4),
            "saved":       saved,
        })

    # ── Save val results ───────────────────────────────────
    print(f"\n💾 Saving training results...")
    save_results(epoch_logs, best_report, best_macro_f1, best_weighted_f1,
                 split_name="val")

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val Macro F1  : {best_macro_f1:.4f}")
    print(f"  Best Val Weighted  : {best_weighted_f1:.4f}")
    print(f"  Model saved to     : {D1_MODEL_DIR}")
    print(f"  Test split saved   : {test_path}")
    print(f"  → Run evaluate.py once on test split for honest final number")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()