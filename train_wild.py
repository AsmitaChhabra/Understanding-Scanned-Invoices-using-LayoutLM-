import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMTokenizerFast,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "models/layoutlm_wild_final"
# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH  = "Into_the_wild_D2/dataset_mapped.json"

NUM_EPOCHS    = 3
BATCH_SIZE    = 2
MAX_SEQ_LEN   = 256
LEARNING_RATE = 1e-5
VAL_SPLIT     = 0.15
SEED          = 42

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ─── DEVICE ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU — training will be slower")


# ─── DATASET ──────────────────────────────────────────────────────────────────
class InvoiceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len=512):
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


# ─── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            bbox           = batch["bbox"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                labels=labels,
            )

            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

    present_labels = sorted(set(all_labels))
    target_names = [ID2LABEL[i] for i in present_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
)
    return report["weighted avg"]["f1-score"], report


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n📂 Loading dataset: {DATASET_PATH}")
    with open(DATASET_PATH, "r") as f:
        all_data = json.load(f)
    print(f"   Total samples: {len(all_data)}")

    train_data, val_data = train_test_split(
        all_data, test_size=VAL_SPLIT, random_state=SEED
    )
    print(f"   Train: {len(train_data)}  |  Val: {len(val_data)}")

    print("\n🤖 Loading LayoutLM tokenizer and model...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained("models/layoutlm_D1_final")
    model     = LayoutLMForTokenClassification.from_pretrained(
        "models/layoutlm_D1_final",
        num_labels=NUM_LABELS,
    )
    model.to(device)

    train_dataset = InvoiceDataset(train_data, tokenizer, MAX_SEQ_LEN)
    val_dataset   = InvoiceDataset(val_data,   tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            bbox           = batch["bbox"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_f1, report = evaluate(model, val_loader)

        print(f"\nEpoch {epoch} — Loss: {avg_loss:.4f}  |  Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"   💾 Best model saved (F1: {best_f1:.4f})")

        if epoch % 2 == 0:
            print("\n   Per-class F1:")
            for label_name, metrics in report.items():
                if label_name in LABEL_MAP:
                    print(f"   {label_name:<20} F1: {metrics['f1-score']:.3f}")

    print(f"\n✅ Training complete!")
    print(f"   Best Val F1 : {best_f1:.4f}")
    print(f"   Model saved : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()