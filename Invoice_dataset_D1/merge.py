import json

with open("Invoice_dataset_D1/preprocessed/batch1_1/preprocessed_D1_1.json") as f:
    batch1 = json.load(f)

with open("Invoice_dataset_D1/preprocessed/batch1_2/preprocessed_D1_2.json") as f:
    batch2 = json.load(f)

with open("Invoice_dataset_D1/preprocessed/batch1_3/preprocessed_D1_3.json") as f:
    batch3 = json.load(f)

merged = batch1 + batch2 + batch3
print(len(merged))  # should be ~1500

with open("Invoice_dataset_D1/preprocessed/preprocessed_dataset.json", "w") as f:
    json.dump(merged, f, indent=2)

#these are relative paths, and they are correct as per the current directory structure. Please ensure that the paths are correct when running the code.
