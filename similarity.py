import json
import os
import numpy as np
import torch
from transformers import LayoutLMModel, LayoutLMTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────
TRAINED_MODEL_PATH = "models/layoutlm20260418_103511" # or models/layoutlm_wild_final for D1 
DATASET_JSON       = "Into_the_wild_D2/dataset_mapped.json" # or Invoice_dataset_D1/preprocessed/batch1_1/dataset.json for D1
OUTPUT_DIR         = "similarity_results_1"

TOP_K               = 5
SIMILARITY_THRESHOLD = 0.80

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DEVICE ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# ─── LOAD MODEL ─────────────────────────────────────────
print("📦 Loading model...")

tokenizer = LayoutLMTokenizerFast.from_pretrained(
    TRAINED_MODEL_PATH,
    local_files_only=True
)

model = LayoutLMModel.from_pretrained(
    TRAINED_MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

# ─── JACCARD SIMILARITY ─────────────────────────────────
def jaccard_similarity(words_a, words_b):
    set_a = set(w.lower() for w in words_a)
    set_b = set(w.lower() for w in words_b)

    if not (set_a | set_b):
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)

# ─── FEATURE EXTRACTION ─────────────────────────────────
def extract_features(sample):
    words = sample["words"]
    boxes = sample["bboxes"]

    # Clamp bbox
    boxes = [[max(0, min(1000, c)) for c in box] for box in boxes]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # ✅ GET word_ids BEFORE moving to device
    word_ids = encoding.word_ids(batch_index=0)

    # NOW move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    aligned_boxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_idx])

    bbox_tensor = torch.tensor([aligned_boxes], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            bbox=bbox_tensor
        )

    # 🔥 mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.detach().cpu().numpy()

# ─── LOAD DATA ──────────────────────────────────────────
print("\n🔍 Extracting features from all invoices...")

with open(DATASET_JSON, "r") as f:
    dataset = json.load(f)

all_features   = []
all_words      = []
all_file_names = []

for sample in tqdm(dataset, desc="Extracting"):
    features = extract_features(sample)

    all_features.append(features)
    all_words.append(sample["words"])
    all_file_names.append(
        sample.get("file_name", f"invoice_{len(all_file_names)}")
    )

feature_matrix = np.vstack([f.squeeze(0) for f in all_features])

print(f"✅ Extracted features for {len(all_features)} invoices")

# ─── COSINE SIMILARITY ─────────────────────────────────
print("\n📐 Computing cosine similarity matrix...")
cosine_matrix = cosine_similarity(feature_matrix)

# ─── JACCARD SIMILARITY ─────────────────────────────────
print("📐 Computing Jaccard similarity matrix...")

n = len(all_words)
jaccard_matrix = np.zeros((n, n))

for i in tqdm(range(n), desc="Jaccard"):
    for j in range(n):
        if i == j:
            jaccard_matrix[i][j] = 1.0
        elif j > i:
            score = jaccard_similarity(all_words[i], all_words[j])
            jaccard_matrix[i][j] = score
            jaccard_matrix[j][i] = score

# ─── TOP-K SIMILARITY ──────────────────────────────────
print(f"\n🔎 Finding top {TOP_K} similar invoices...")

results = []

for i in range(n):
    cosine_scores  = cosine_matrix[i]
    jaccard_scores = jaccard_matrix[i]

    top_k_cosine = sorted(
        [(j, float(cosine_scores[j])) for j in range(n) if j != i],
        key=lambda x: x[1], reverse=True
    )[:TOP_K]

    top_k_jaccard = sorted(
        [(j, float(jaccard_scores[j])) for j in range(n) if j != i],
        key=lambda x: x[1], reverse=True
    )[:TOP_K]

    result = {
        "query_invoice": all_file_names[i],
        "top_similar_cosine": [
            {
                "invoice": all_file_names[j],
                "score": round(score, 4),
                "similar": score >= SIMILARITY_THRESHOLD
            }
            for j, score in top_k_cosine
        ],
        "top_similar_jaccard": [
            {
                "invoice": all_file_names[j],
                "score": round(score, 4),
                "similar": score >= SIMILARITY_THRESHOLD
            }
            for j, score in top_k_jaccard
        ]
    }

    results.append(result)

# ─── SAVE RESULTS ───────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "similarity_results.json")

with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ─── PRINT SAMPLE ───────────────────────────────────────
print(f"\n✅ Similarity detection complete!")
print(f"   Results saved: {out_path}")

print(f"\n📋 Sample result:")
print(f"   Query: {results[0]['query_invoice']}")

print(f"\n   Top {TOP_K} (Cosine):")
for match in results[0]["top_similar_cosine"]:
    flag = "✅" if match["similar"] else "❌"
    print(f"   {match['invoice']} | {match['score']} {flag}")

print(f"\n   Top {TOP_K} (Jaccard):")
for match in results[0]["top_similar_jaccard"]:
    flag = "✅" if match["similar"] else "❌"
    print(f"   {match['invoice']} | {match['score']} {flag}")

# ─── SUMMARY ────────────────────────────────────────────
cosine_similar_pairs = sum(
    1 for r in results for m in r["top_similar_cosine"] if m["similar"]
)

jaccard_similar_pairs = sum(
    1 for r in results for m in r["top_similar_jaccard"] if m["similar"]
)

print(f"\n📊 Summary:")
print(f"   Total invoices: {n}")
print(f"   Cosine similar pairs: {cosine_similar_pairs}")
print(f"   Jaccard similar pairs: {jaccard_similar_pairs}")