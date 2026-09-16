from datasets import load_dataset

# 1. LOAD THE DATASETS
print("Loading HaluEval...")
halueval = load_dataset("flowaicom/HaluEval")

print("Loading TruthfulQA...")
truthfulqa = load_dataset("truthfulqa/truthful_qa", "generation")

print("Loading FEVER...")
fever_urls = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "validation": "https://fever.ai/download/fever/shared_task_dev.jsonl"
}
fever = load_dataset("json", data_files=fever_urls)

# 2. DEFINE THE UNIFIED LABEL MAPPINGS
# This fulfills Member 1's responsibility to map everything to a binary format[cite: 1]
def map_halueval(row):
    # Note: We might need to tweak 'Hallucinated' if the dataset uses 'Yes'/'No' instead
    is_halluc = 1 if row['label'] == 'Hallucinated' else 0 
    return {"text": row['answer'], "is_hallucinated": is_halluc}

def map_fever(row):
    # 'SUPPORTS' is faithful (0). Both 'REFUTES' and 'NOT ENOUGH INFO' count as hallucinations (1)[cite: 1].
    is_halluc = 0 if row['label'] == 'SUPPORTS' else 1
    return {"text": row['claim'], "is_hallucinated": is_halluc}

def map_truthfulqa(row):
    # For now, grabbing the best answer as a faithful baseline (0)
    return {"text": row['best_answer'], "is_hallucinated": 0}

# 3. APPLY THE MAPPINGS TO THE DATASETS
print("Applying unified label schema to datasets...")
halueval = halueval.map(map_halueval)
fever = fever.map(map_fever)
truthfulqa = truthfulqa.map(map_truthfulqa)

# 4. VERIFY THE RESULTS
print("\n--- Success! Here is a mapped sample from HaluEval ---")
# Printing the first row to ensure our 'text' and 'is_hallucinated' columns were created
print(halueval['test'][0])

# Print the first row of each dataset to inspect the actual label strings
print("\nRaw HaluEval Row:", halueval['test'][0])
print("\nRaw FEVER Row:", fever['train'][0])
print("\nRaw TruthfulQA Row:", truthfulqa['validation'][0])