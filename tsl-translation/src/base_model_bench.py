import torch, sys, os, json, time, re, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
import evaluate as eval_lib
from tqdm import tqdm

# --- [ 1. PATH AND COMPATIBILITY SETUP ] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

for bit in range(1, 17):
    if not hasattr(torch, f"int{bit}"): setattr(torch, f"int{bit}", torch.int8)
os.environ["UNSLOTH_ENABLE_PATCHES"] = "0"

# --- [ 2. HELPER FUNCTIONS ] ---
def turkish_lower(text):
    if not text: return ""
    return text.replace('İ', 'i').replace('I', 'ı').lower()

def turkish_capitalize(text):
    if not text: return ""
    text = text.strip().lower()
    first = text[0]
    if first == 'i': res = 'İ' + text[1:]
    elif first == 'ı': res = 'I' + text[1:]
    else: res = first.upper() + text[1:]
    return res

def polish_turkish(text):
    text = text.strip()
    if not text: return ""
    text = turkish_capitalize(text)
    def cap_match(match): return match.group(1) + turkish_capitalize(match.group(2))
    text = re.sub(r'([.!?]\s+)([a-zığüşöç])', cap_match, text)
    def cap_proper(match): return turkish_capitalize(match.group(0))
    text = re.sub(r"\b[a-zığüşöç]+'[a-zığüşöç]*\b", cap_proper, text)
    if not text[-1] in ".!?": text += "."
    return text

def get_chat_template(model_name, instruction, gloss):
    m = model_name.lower()
    if "gemma" in m:
        return f"<start_of_turn>user\n{instruction}\n\nGloss: {gloss}<end_of_turn>\n<start_of_turn>model\n"
    elif "llama" in m:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nGloss: {gloss}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else: # Trendyol
        return f"<|im_start|>user\n{instruction}\n\nGloss: {gloss}<|im_end|>\n<|im_start|>assistant\n"

# --- [ 3. INFERENCE ENGINE ] ---
def run_inference(model, tokenizer, model_id, strategy_key, output_dir, valid_samples, chrf):
    from config.prompts import PROMPT_STRATEGIES
    instruction = PROMPT_STRATEGIES[strategy_key]
    
    res_path = os.path.join(output_dir, f"result_{strategy_key.lower()}.json")
    
    # --- RESUME AND INCREMENTAL LOGIC ---
    results = []
    processed_glosses = set()
    if os.path.exists(res_path):
        with open(res_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                processed_glosses = {r['gloss'] for r in results}
                print(f"[INFO] Resuming {strategy_key}: Found {len(results)} existing entries.")
            except:
                results = []

    print(f"\n[INFO] Running Inference: {model_id} | {strategy_key}")
    
    for sample in tqdm(valid_samples, desc=f"Eval {strategy_key}"):
        # Skip if already processed
        if sample['input'] in processed_glosses:
            continue

        g_in = turkish_lower(sample['input'])
        prompt = get_chat_template(model_id, instruction, g_in)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, num_beams=5, 
                repetition_penalty=1.15, do_sample=False
            )
        duration = time.perf_counter() - start_time
        
        prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        final_prediction = polish_turkish(prediction)
        
        # Compute chrF++ score
        score = chrf.compute(predictions=[final_prediction], references=[[sample["output"]]], word_order=2)["score"]
        
        results.append({
            "gloss": sample['input'], 
            "expected": sample["output"], 
            "prediction": final_prediction, 
            "latency_sec": round(duration, 4),
            "chrf": round(score, 2)
        })

        # Incremental save: write results after each sample
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"[DONE] Strategy {strategy_key} completed.")

# --- [ 4. MAIN ] ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()

    model_slug = args.model_id.split("/")[-1].lower().replace("-", "_")
    output_dir = f"experiments/baselines/{model_slug}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer & Model
    if "trendyol" in args.model_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.model_id, legacy=False, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.eval()

    # 2. Data & Metric
    chrf = eval_lib.load("chrf")
    valid_path = os.path.join(project_root, "data/processed/valid.jsonl")
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_samples = [json.loads(line) for line in f]

    # Strategy mapping per model
    MODEL_STRATEGY_MAP = {
        "gemma": ("P3_EN", "P3_TR"),
        "llama": ("P2_EN", "P2_TR"),
        "trendyol": ("P2_EN", "P2_TR")
    }
    
    key = "gemma"
    if "llama" in args.model_id.lower(): key = "llama"
    elif "trendyol" in args.model_id.lower(): key = "trendyol"
    
    en_strat, tr_strat = MODEL_STRATEGY_MAP[key]

    # Run Dual Language Inference
    run_inference(model, tokenizer, args.model_id, en_strat, output_dir, valid_samples, chrf)
    run_inference(model, tokenizer, args.model_id, tr_strat, output_dir, valid_samples, chrf)

if __name__ == "__main__":
    main()