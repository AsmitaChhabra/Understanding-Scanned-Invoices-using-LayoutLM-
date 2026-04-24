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

from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForTokenClassification,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================================================
# CONFIG
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_PATH = "Into_the_wild_D2/dataset_mapped.json"

OUTPUT_DIR = "models/layoutlm_wild_final"

NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LEN = 256
LEARNING_RATE = 1e-5
VAL_SPLIT = 0.15
SEED = 42

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

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(SEED)


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

print(f"✅ Using {device_name}")


# =========================================================
# DATASET
# =========================================================

class InvoiceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        words = sample["words"]
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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "bbox": torch.tensor(aligned_bboxes, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                bbox=batch["bbox"].to(device),
            )

            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)

            for pred_seq, label_seq, conf_seq in zip(
                preds,
                batch["labels"],
                confidence
            ):
                for p, l, c in zip(
                    pred_seq.tolist(),
                    label_seq.tolist(),
                    conf_seq.tolist()
                ):
                    if l != -100 and c > 0.6:
                        all_preds.append(p)
                        all_labels.append(l)

    if not all_labels:
        return 0.0, {}

    present_labels = sorted(set(all_labels))

    report = classification_report(
        all_labels,
        all_preds,
        labels=present_labels,
        target_names=[ID2LABEL[i] for i in present_labels],
        zero_division=0,
        output_dict=True
    )

    weighted_f1 = report["weighted avg"]["f1-score"]

    return weighted_f1, report


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(epoch_logs, final_report, best_f1):
    csv_path = os.path.join(RESULTS_DIR, "training_log.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_f1", "saved"]
        )
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"📊 Training log saved: {csv_path}")

    txt_path = os.path.join(RESULTS_DIR, "summary.txt")

    with open(txt_path, "w") as f:
        f.write("====================================\n")
        f.write("LAYOUTLM TRAINING SUMMARY\n")
        f.write("====================================\n\n")

        f.write(f"Device: {device_name}\n")
        f.write(f"Best Validation F1: {best_f1:.4f}\n\n")

        f.write("Per-Class Metrics\n")
        f.write("------------------------------------\n")

        for label_name, metrics in final_report.items():
            if isinstance(metrics, dict) and label_name not in [
                "accuracy",
                "macro avg",
                "weighted avg"
            ]:
                f.write(
                    f"{label_name}\n"
                    f"Precision: {metrics['precision']:.4f}\n"
                    f"Recall: {metrics['recall']:.4f}\n"
                    f"F1: {metrics['f1-score']:.4f}\n"
                    f"Support: {int(metrics['support'])}\n\n"
                )

    print(f"📄 Summary saved: {txt_path}")


# =========================================================
# TOTAL EXTRACTION
# =========================================================

def extract_total_from_predictions(words, preds):
    candidates = []

    for word, label in zip(words, preds):
        try:
            value = float(word.replace(",", ""))

            if label == 9 and value < 1e7:
                candidates.append(value)

        except:
            continue

    if candidates:
        return max(candidates)

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
# MAIN
# =========================================================

def main():
    print(f"\n📂 Loading dataset: {DATASET_PATH}")

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    train_data, val_data = train_test_split(
        data,
        test_size=VAL_SPLIT,
        random_state=SEED
    )

    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")

    print("\n🤖 Loading LayoutLM model...")

    tokenizer = LayoutLMTokenizerFast.from_pretrained(
        "models/layoutlm_D1_final"
    )

    model = LayoutLMForTokenClassification.from_pretrained(
        "models/layoutlm_D1_final",
        num_labels=NUM_LABELS,
    ).to(device)

    train_loader = DataLoader(
        InvoiceDataset(train_data, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        InvoiceDataset(val_data, tokenizer),
        batch_size=BATCH_SIZE
    )

    counts = Counter(
        [label for sample in train_data for label in sample["labels"]]
    )

    total = sum(counts.values())

    weights = torch.tensor(
        [
            np.log(1 + total / counts[i]) if counts[i] > 0 else 1.0
            for i in range(NUM_LABELS)
        ],
        dtype=torch.float32
    ).to(device)

    loss_fct = torch.nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=-100,
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = 0.0
    final_report = {}
    epoch_logs = []

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                bbox=batch["bbox"].to(device),
            )

            loss = loss_fct(
                outputs.logits.view(-1, NUM_LABELS),
                batch["labels"].to(device).view(-1)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = round(total_loss / len(train_loader), 4)

        val_f1, report = evaluate(model, val_loader)
        val_f1 = round(val_f1, 4)

        saved = False

        print(
            f"\nEpoch {epoch} | "
            f"Loss: {avg_loss} | "
            f"Val F1: {val_f1}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            final_report = report
            saved = True

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            print(f"💾 Best model saved to:")
            print(OUTPUT_DIR)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_f1": val_f1,
            "saved": saved
        })

    print("\n🧪 Testing extraction...\n")

    for sample in val_data[:5]:
        pred_total = extract_total_from_predictions(
            sample["words"],
            sample["labels"]
        )

        print(f"Predicted Total: {pred_total}")

    print("\n💾 Saving results...")

    save_results(
        epoch_logs=epoch_logs,
        final_report=final_report,
        best_f1=best_f1
    )

    print("\n✅ Training Complete!")
    print(f"Best F1 Score: {best_f1}")
    print(f"Model Folder: {OUTPUT_DIR}")
    print(f"Results Folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
