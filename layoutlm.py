import os
import json
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "models/layoutlm_D1_final"

# ─── CONFIG ─────────────────────────────────────────────
DATASET_PATH  = "Invoice_dataset_D1/preprocessed/batch1_1/dataset.json"

NUM_EPOCHS    = 2
BATCH_SIZE    = 2
MAX_SEQ_LEN   = 128
LEARNING_RATE = 3e-5
VAL_SPLIT     = 0.15
SEED          = 42
PATIENCE      = 1

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

device = torch.device("cpu")
print(f"Using device: {device}")

# ─── AUGMENTATION ───────────────────────────────────────
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

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def add_token_noise(word):
    if random.random() < 0.1:
        return word.replace("0", "O").replace("1", "I")
    return word

# ─── DATASET ────────────────────────────────────────────
class InvoiceDataset(Dataset):
    def __init__(self, samples, tokenizer, augment=False):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.augment   = augment

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
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
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

# ─── EVALUATION (with confidence filtering) ──────────────
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
            )

            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)

            for p_seq, l_seq, c_seq in zip(preds, labels, confidence):
                for p, l, c in zip(p_seq.tolist(), l_seq.tolist(), c_seq.tolist()):
                    if l != -100 and c > 0.6:  # 🔥 confidence filtering
                        all_preds.append(p)
                        all_labels.append(l)

    if len(all_labels) == 0:
        return 0.0

    labels_present = sorted(set(all_labels))

    report = classification_report(
        all_labels,
        all_preds,
        labels=labels_present,
        target_names=[ID2LABEL[i] for i in labels_present],
        zero_division=0,
        output_dict=True
    )

    return report["macro avg"]["f1-score"]

# ─── MAIN ───────────────────────────────────────────────
def main():
    print("Loading dataset...")
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    train_data, val_data = train_test_split(data, test_size=VAL_SPLIT, random_state=SEED)

    tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=NUM_LABELS
    )

    # 🔥 Unfreeze 2 layers instead of 1
    for name, param in model.named_parameters():
        if "classifier" in name or "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    model.to(device)

    train_dataset = InvoiceDataset(train_data, tokenizer, augment=True)
    val_dataset   = InvoiceDataset(val_data, tokenizer, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ─── BETTER CLASS WEIGHTS ─────────────────────────
    all_labels_flat = [l for s in train_data for l in s["labels"]]
    counts = Counter(all_labels_flat)
    total = sum(counts.values())

    class_weights_list = []
    for i in range(NUM_LABELS):
        if counts[i] == 0:
            class_weights_list.append(1.0)
        else:
            class_weights_list.append(np.log(1 + total / counts[i]))

    class_weights = torch.tensor(class_weights_list, dtype=torch.float).to(device)

    # 🔥 Label smoothing added
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-100,
        label_smoothing=0.1
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    best_f1 = 0
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
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
            )

            logits = outputs.logits
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        train_f1 = evaluate(model, train_loader)
        val_f1   = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("✅ Model saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early stopping")
                break

    print("\nTraining complete.")

if __name__ == "__main__":
    main()