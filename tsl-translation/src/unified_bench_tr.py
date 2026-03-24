import torch, sys, os, json, time, re, argparse, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import evaluate as eval_lib
from tqdm import tqdm

# Ensure local config imports resolve regardless of launch directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- [ 1. ARMOR PATCH ] ---
for bit in range(1, 17):
    if not hasattr(torch, f"int{bit}"): setattr(torch, f"int{bit}", torch.int8)
os.environ["UNSLOTH_ENABLE_PATCHES"] = "0"

def turkish_lower(text):
    if not text: return ""
    return text.replace('İ', 'i').replace('I', 'ı').lower()

def turkish_capitalize(text):
    if not text: return ""
    first = text[0]
    if first == 'i': return 'İ' + text[1:]
    if first == 'ı': return 'I' + text[1:]
    return first.upper() + text[1:]

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

def get_chat_template(model_name, instruction, gloss, output=""):
    m = model_name.lower()
    if "gemma" in m:
        return f"<start_of_turn>user\n{instruction}\n\nGloss: {gloss}<end_of_turn>\n<start_of_turn>model\n{output}"
    elif "llama" in m:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nGloss: {gloss}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}"
    else: # Trendyol
        return f"<|im_start|>user\n{instruction}\n\nGloss: {gloss}<|im_end|>\n<|im_start|>assistant\n{output}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    args = parser.parse_args()

    model_slug = args.model_id.split("/")[-1].lower().replace("-", "_")
    output_dir = f"experiments/benchmarks/{model_slug}/{args.strategy.lower()}"
    adapter_path = os.path.join(output_dir, "final_adapter")
    result_path = os.path.join(output_dir, "result.json")

    # Skip if result already exists
    if os.path.exists(result_path):
        print(f"[SKIP] {result_path} exists.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer & Model (SDPA + BFloat16)
    is_trendyol = "trendyol" in args.model_id.lower()
    if is_trendyol:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_id, legacy=False, use_fast=False)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if is_trendyol: base_model.resize_token_embeddings(len(tokenizer))

    # --- Skip training if adapter already exists, proceed to evaluation ---
    if os.path.exists(adapter_path):
        print(f"[INFO] Adapter found at {adapter_path}. Skipping training, moving to EVAL.")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"[INFO] Starting FULL TRAINING for {args.model_id}...")
        base_model = prepare_model_for_kbit_training(base_model)
        model = get_peft_model(base_model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
        
        from config.prompts import PROMPT_STRATEGIES
        instruction = PROMPT_STRATEGIES[args.strategy]

        dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl", "test": "data/processed/valid.jsonl"})
        def map_fn(x): return {"text": get_chat_template(args.model_id, instruction, turkish_lower(x['input']), turkish_lower(x['output']) + tokenizer.eos_token)}
        
        trainer = SFTTrainer(
            model=model, train_dataset=dataset["train"].map(map_fn), eval_dataset=dataset["test"].map(map_fn),
            args=SFTConfig(
                output_dir=output_dir, dataset_text_field="text", max_seq_length=1024,
                per_device_train_batch_size=8, gradient_accumulation_steps=2,
                learning_rate=1e-4, num_train_epochs=5, bf16=True, save_total_limit=1,
                load_best_model_at_end=True, eval_strategy="steps", eval_steps=100, report_to="none"
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

    # 4. Evaluation (using autocast to handle dtype compatibility)
    print(f"[EVAL] Starting Evaluation for {model_slug}...")
    chrf = eval_lib.load("chrf")
    test_dataset = load_dataset("json", data_files={"test": "data/processed/valid.jsonl"})["test"]
    
    from config.prompts import PROMPT_STRATEGIES
    instruction = PROMPT_STRATEGIES[args.strategy]

    model.eval()
    results = []
    total_latency = 0
    
    for sample in tqdm(test_dataset, desc="TR-Eval"):
        g_in = turkish_lower(sample['input'])
        prompt = get_chat_template(args.model_id, instruction, g_in)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start = time.perf_counter()
        with torch.no_grad():
            # Use autocast to resolve dtype mismatch
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(**inputs, max_new_tokens=128, num_beams=5, repetition_penalty=1.15, do_sample=False)
        
        dur = time.perf_counter() - start
        total_latency += dur
        prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        final_pred = polish_turkish(prediction)
        
        score = chrf.compute(predictions=[final_pred], references=[[sample["output"]]], word_order=2)["score"]
        results.append({"gloss": sample['input'], "expected": sample["output"], "prediction": final_pred, "latency_sec": round(dur, 4), "chrf": round(score, 2)})

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[SUCCESS] {model_slug} Complete. Avg Latency: {total_latency/len(test_dataset):.4f}s")
    
if __name__ == "__main__":
    main()