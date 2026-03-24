"""
SLR Project: Final Comparison Judge (1-10 Scoring)
Purpose: Evaluates all models trained on HPC against previous baselines.
Location: RUN THIS LOCALLY (Mac/Windows) to avoid internet issues on HPC.
"""

import os
import json
import time
import re
import google.generativeai as genai
from tqdm import tqdm

# --- 1. Configuration ---
# Load .env file from project root if it exists
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or export it in your shell.")
MODEL_NAME = "gemini-2.5-flash" # Use Flash for speed and cost-efficiency
SUCCESS_THRESHOLD = 5

# --- 2. Target Files Matrix (New HPC Structure + Old Champions) ---
# Paths to result files in the 'experiments' directory.
TARGET_FILES = [
    # --- NEW HPC MODELS (Benchmark Matrix) ---

    "experiments/baselines/gemma_2_9b_it/result_p3_en.json",
    "experiments/baselines/gemma_2_9b_it/result_p3_tr.json",

    "experiments/baselines/meta_llama_3.1_8b_instruct/result_p2_en.json",
    "experiments/baselines/meta_llama_3.1_8b_instruct/result_p2_tr.json",

    "experiments/baselines/trendyol_llm_7b_chat_v1.8/result_p2_en.json",
    "experiments/baselines/trendyol_llm_7b_chat_v1.8/result_p2_tr.json",


    
    "experiments/benchmarks/gemma_2_9b_it/p1_en/result.json",
    "experiments/benchmarks/gemma_2_9b_it/p2_en/result.json",
    "experiments/benchmarks/gemma_2_9b_it/p3_en/result.json",

    "experiments/benchmarks/meta_llama_3.1_8b_instruct/p1_en/result.json",
    "experiments/benchmarks/meta_llama_3.1_8b_instruct/p2_en/result.json",
    "experiments/benchmarks/meta_llama_3.1_8b_instruct/p3_en/result.json",
    
    "experiments/benchmarks/trendyol_llm_7b_chat_v1.8/p1_en/result.json",
    "experiments/benchmarks/trendyol_llm_7b_chat_v1.8/p2_en/result.json",
    "experiments/benchmarks/trendyol_llm_7b_chat_v1.8/p3_en/result.json",

 
    "experiments/benchmarks/gemini_pro_p3_en/results.json",

    "experiments/benchmarks/gemma_2_9b_it/p3_en_rag/result.json",

    "experiments/benchmarks/gemma_2_9b_it/p3_tr/result.json",
    "experiments/benchmarks/trendyol_llm_7b_chat_v1.8/p2_tr/result.json",
    "experiments/benchmarks/meta_llama_3.1_8b_instruct/p2_tr/result.json",

]

genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel(MODEL_NAME)

def get_detailed_score(gloss, expected, predicted):
    """Expert prompting for TSL linguistics assessment."""
    prompt = f"""
[Expert Identity]: You are a linguistics professor specializing in Turkish Sign Language (TSL) and Turkish translation.

[Task]: Evaluate the quality of the "Model Prediction" compared to the "Ground Truth", using the "Input Gloss" for context.

[Data]:
- Gloss: "{gloss}"
- Ground Truth: "{expected}"
- Model Prediction: "{predicted}"

[Scoring Rubric (1-10)]:
- 10: Perfect translation. Natural, accurate, and grammatically flawless.
- 8-9: Excellent. Accurate meaning, very minor stylistic or punctuation issues.
- 6-7: Good/Acceptable. Meaning is clear, but has slight grammatical errors or awkward phrasing.
- 4-5: Poor. Meaning is partially preserved but difficult to understand.
- 1-3: Failure. Gibberish or completely wrong meaning.

[Rules]: Focus on Semantic Preservation. Return ONLY the integer score.
"""
    try:
        response = judge_model.generate_content(prompt)
        res_text = response.text.strip()
        match = re.search(r'\b([1-9]|10)\b', res_text)
        return int(match.group(0)) if match else None
    except Exception as e:
        print(f"\n[ERROR] API Call failed: {e}")
        return None

def process_file_refined(file_path):
    if not os.path.exists(file_path):
        # Skip if the result file does not exist yet
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build descriptive label for reporting
    folder_parts = file_path.split('/')
    if 'benchmarks' in folder_parts:
        model_name = folder_parts[-3].replace('_', ' ').title()
        strategy = folder_parts[-2].upper()
        label = f"{model_name} | {strategy}"
    else:
        label = folder_parts[-2] # Old structure

    print(f"\n[JUDGE] Assessing: {label}")
    
    updated = 0
    scores = []
    
    for entry in tqdm(data, desc=f"Scoring Samples"):
        # --- RESUME LOGIC ---
        # Skip samples that have already been scored.
        if "gemini_score_10" in entry and entry["gemini_score_10"] is not None:
            scores.append(entry["gemini_score_10"])
            continue
            
        score = get_detailed_score(entry["gloss"], entry["expected"], entry["prediction"])
        
        if score is not None:
            entry["gemini_score_10"] = score
            scores.append(score)
            updated += 1
            
            # Checkpoint: save every 10 samples to prevent data loss
            if updated % 10 == 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
        
        time.sleep(0.5) # Rate limit protection

    # Final Save for this file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"[INFO] Done! Avg Score: {avg:.2f}")

def generate_final_table():
    """Aggregates all result.json files into a comparison table."""
    final_report = {}
    for file_path in TARGET_FILES:
        if not os.path.exists(file_path): continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Calculate Metrics
        scores = [i["gemini_score_10"] for i in data if "gemini_score_10" in i]
        if not scores: continue
        
        success_rate = (len([s for s in scores if s >= SUCCESS_THRESHOLD]) / len(data)) * 100
        avg_chrf = sum(i.get("chrf", 0) for i in data) / len(data)
        
        # Descriptive Key
        key = file_path.replace("experiments/benchmarks/", "").replace("/result.json", "").replace("/", "_")
        
        final_report[key] = {
            "Avg Score": round(sum(scores)/len(scores), 2),
            "Success Rate (%)": round(success_rate, 2),
            "Avg chrF": round(avg_chrf, 2),
            "Samples": len(data)
        }

    with open("experiments/REPORT.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    
    print("\n" + "="*50)
    print("FINAL REPORT GENERATED: experiments/REPORT.json")
    print("="*50)

if __name__ == "__main__":
    for target in TARGET_FILES:
        process_file_refined(target)
    generate_final_table()