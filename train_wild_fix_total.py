import os
import json
import csv
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
from sklearn.metrics import classification_report

from config_D2 import (
    SEED,
    LABEL_MAP,
    ID2LABEL,
    NUM_LABELS,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    VAL_SPLIT,
    D2_LEARNING_RATE,
    D2_NUM_EPOCHS,
    D1_MODEL_DIR,
    D2_MODEL_DIR,
    D2_MAPPED_JSON,
    RESULTS_DIR_D2,
)

# =========================================================
# TRAINING HYPERPARAMETERS
# D2-specific tuning knobs — not shared constants
# =========================================================

WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
LABEL_SMOOTHING = 0.1
PATIENCE        = 3

# For transfer learning on only 79 samples, unfreeze fewer
# layers than D1 — just the classifier and top layer (11)
# to avoid catastrophic forgetting on a tiny dataset
UNFREEZE_LAYERS = [11]

# =========================================================
# SETUP
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(D2_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_D2, exist_ok=True)

torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA GPU"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"✅ Using {device_name}")

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
# EVALUATE — no confidence filtering
# Confidence filtering is a deployment decision only.
# See DEPLOY_CONFIDENCE_THRESHOLD in config.py.
# =========================================================

def evaluate(model, loader):
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

    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=[ID2LABEL[i] for i in present_labels],
        zero_division=0,
        output_dict=True,
    )

    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    return macro_f1, weighted_f1, report

# =========================================================
# TOTAL EXTRACTION — rule-based fallback
# Uses label ID from config (total = 8), not hardcoded 9
# =========================================================

def extract_total_from_predictions(words, preds):
    total_label_id = LABEL_MAP["total"]   # 8 — from config, never hardcoded
    candidates     = []

    # First pass — model-predicted total tokens
    for word, label in zip(words, preds):
        try:
            value = float(word.replace(",", ""))
            if label == total_label_id and value < 1e7:
                candidates.append(value)
        except:
            continue

    if candidates:
        return max(candidates)

    # Second pass — keyword fallback
    keywords = ["TOTAL", "AMOUNT", "PAYABLE", "GRAND"]
    for i, word in enumerate(words):
        if word.upper() in keywords:
            for j in range(i + 1, min(i + 5, len(words))):
                try:
                    value = float(words[j].replace(",", ""))
                    if value < 1e7:
                        candidates.append(value)
                except:
                    continue

    if candidates:
        return max(candidates)

    # Last resort — largest number on the page
    all_numbers = []
    for word in words:
        try:
            value = float(word.replace(",", ""))
            if value < 1e7:
                all_numbers.append(value)
        except:
            continue

    return max(all_numbers) if all_numbers else None

# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(epoch_logs, final_report, best_macro_f1, best_weighted_f1):
    csv_path = os.path.join(RESULTS_DIR_D2, f"training_log_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "macro_f1", "weighted_f1", "saved"
        ])
        writer.writeheader()
        writer.writerows(epoch_logs)
    print(f"   📊 Training log : {csv_path}")

    txt_path = os.path.join(RESULTS_DIR_D2, f"summary_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write("LAYOUTLM D2 TRANSFER LEARNING SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Timestamp        : {timestamp}\n")
        f.write(f"Device           : {device_name}\n")
        f.write(f"Source model     : {D1_MODEL_DIR}\n")
        f.write(f"Dataset          : {D2_MAPPED_JSON}\n")
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
    print(f"  LAYOUTLM D2 TRANSFER LEARNING")
    print(f"  Source: {D1_MODEL_DIR}")
    print(f"{'='*55}\n")

    # ── Load data ──────────────────────────────────────────
    print(f"📂 Loading: {D2_MAPPED_JSON}")
    with open(D2_MAPPED_JSON, "r") as f:
        data = json.load(f)
    print(f"   Total samples: {len(data)}")

    # ── Split ──────────────────────────────────────────────
    train_data, val_data = train_test_split(
        data, test_size=VAL_SPLIT, random_state=SEED
    )
    print(f"   Train : {len(train_data)}")
    print(f"   Val   : {len(val_data)}")

    # ── Load tokenizer + model from D1 checkpoint ──────────
    print(f"\n🤖 Loading D1 checkpoint: {D1_MODEL_DIR}")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(D1_MODEL_DIR)
    model     = LayoutLMForTokenClassification.from_pretrained(
        D1_MODEL_DIR,
        num_labels=NUM_LABELS,
    )

    # ── Freeze all, selectively unfreeze ───────────────────
    # 79 samples — only unfreeze top layer + classifier
    # to avoid catastrophic forgetting
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        for layer_num in UNFREEZE_LAYERS:
            if f"encoder.layer.{layer_num}" in name:
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params : {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")
    print(f"   Unfrozen         : layer {UNFREEZE_LAYERS} + classifier")

    model.to(device)

    # ── Data loaders ───────────────────────────────────────
    train_loader = DataLoader(
        InvoiceDataset(train_data, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        InvoiceDataset(val_data, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ── Class weights ──────────────────────────────────────
    all_labels_flat = [l for s in train_data for l in s["labels"]]
    counts          = Counter(all_labels_flat)
    total_tokens    = sum(counts.values())

    weights = torch.tensor([
        np.log(1 + total_tokens / counts[i]) if counts[i] > 0 else 1.0
        for i in range(NUM_LABELS)
    ], dtype=torch.float32).to(device)

    print(f"\n📊 Label distribution (train):")
    for i in range(NUM_LABELS):
        if counts[i] > 0:
            pct = 100 * counts[i] / total_tokens
            print(f"   {ID2LABEL[i]:<20} count: {counts[i]:6d}  ({pct:.1f}%)  "
                  f"weight: {weights[i].item():.3f}")

    # ── Loss ───────────────────────────────────────────────
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ── Optimizer + scheduler ──────────────────────────────
    optimizer    = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=D2_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps  = len(train_loader) * D2_NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n⚙️  Training config:")
    print(f"   Max epochs    : {D2_NUM_EPOCHS}")
    print(f"   Batch size    : {BATCH_SIZE}")
    print(f"   Learning rate : {D2_LEARNING_RATE}")
    print(f"   Warmup steps  : {warmup_steps} / {total_steps}")
    print(f"   Patience      : {PATIENCE}")

    # ── Training loop ──────────────────────────────────────
    best_macro_f1    = 0.0
    best_weighted_f1 = 0.0
    best_report      = {}
    patience_counter = 0
    epoch_logs       = []

    print(f"\n🚀 Starting training...\n")

    for epoch in range(1, D2_NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{D2_NUM_EPOCHS}"):
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

        train_macro, train_weighted, _      = evaluate(model, train_loader)
        val_macro,   val_weighted,   report = evaluate(model, val_loader)

        gap             = train_macro - val_macro
        overfit_warning = "  ⚠️  Possible overfit" if gap > 0.10 else ""

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
            model.save_pretrained(D2_MODEL_DIR)
            tokenizer.save_pretrained(D2_MODEL_DIR)
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

    # ── Test extraction on val samples ─────────────────────
    print(f"\n🧪 Total extraction check (5 val samples):")
    for sample in val_data[:5]:
        pred_total = extract_total_from_predictions(
            sample["words"], sample["labels"]
        )
        print(f"   Predicted Total: {pred_total}")

    # ── Save results ───────────────────────────────────────
    print(f"\n💾 Saving results...")
    save_results(epoch_logs, best_report, best_macro_f1, best_weighted_f1)

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Val Macro F1  : {best_macro_f1:.4f}")
    print(f"  Best Val Weighted  : {best_weighted_f1:.4f}")
    print(f"  Model saved to     : {D2_MODEL_DIR}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()