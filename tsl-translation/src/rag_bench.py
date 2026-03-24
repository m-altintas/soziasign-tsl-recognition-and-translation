import torch, sys, os, json, time, re, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import evaluate as eval_lib
from tqdm import tqdm

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# --- [ 1. PATH AND COMPATIBILITY SETUP ] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

for bit in range(1, 17):
    if not hasattr(torch, f"int{bit}"): setattr(torch, f"int{bit}", torch.int8)

# --- [ 2. LANGUAGE AND POLISH FUNCTIONS ] ---
def turkish_lower(text):
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

def get_gemma_template(instruction, gloss, context="", output=""):
    """RAG-augmented Gemma template. Context is appended below the instruction."""
    if context:
        full_instruction = f"{instruction}\n\n{context}"
    else:
        full_instruction = instruction
    return f"<start_of_turn>user\n{full_instruction}\n\nGloss: {gloss}<end_of_turn>\n<start_of_turn>model\n{output}"

# --- [ 3. RAG MOTORU ] ---
def main():
    MODEL_ID = "google/gemma-2-9b-it"
    ADAPTER_PATH = "experiments/benchmarks/gemma_2_9b_it/p3_en/final_adapter"
    OUTPUT_DIR = "experiments/benchmarks/gemma_2_9b_it/p3_en_rag"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load SOTA Model & Tokenizer (same memory configuration as main benchmark)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # 2. Embedding Model & Train Pool (retrieval only from training set to prevent leakage)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    train_data = []
    with open(os.path.join(project_root, "data/processed/train.jsonl"), "r", encoding="utf-8") as f:
        for line in f: train_data.append(json.loads(line))
    
    train_glosses = [turkish_lower(item['input']) for item in train_data]
    train_embeddings = embedder.encode(train_glosses, convert_to_tensor=True)

    # 3. Load prompt strategy from config (for consistency with main benchmark)
    from config.prompts import PROMPT_STRATEGIES
    # Original P3_EN strategy used during SOTA model training
    BASE_INSTRUCTION = PROMPT_STRATEGIES["P3_EN"]

    # 4. Inference
    chrf = eval_lib.load("chrf")
    valid_path = os.path.join(project_root, "data/processed/valid.jsonl")
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_samples = [json.loads(line) for line in f]

    results = []
    total_latency = 0

    print(f"[INFO] Starting Scientific RAG Evaluation...")
    for sample in tqdm(valid_samples):
        query_gloss = turkish_lower(sample['input'])
        
        # RAG retrieval (from training set only)
        query_emb = embedder.encode(query_gloss, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, train_embeddings, top_k=2)[0]
        
        context_str = "Reference Examples for Morphology:\n"
        for hit in hits:
            idx = hit['corpus_id']
            context_str += f"- Gloss: {train_data[idx]['input']} -> Turkish: {train_data[idx]['output']}\n"

        # Controlled comparison: original template + RAG context
        prompt = get_gemma_template(BASE_INSTRUCTION, query_gloss, context=context_str)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Controlled comparison: beam search = 5
                outputs = model.generate(**inputs, max_new_tokens=128, num_beams=5, repetition_penalty=1.15, do_sample=False)
        dur = time.perf_counter() - start_time
        total_latency += dur

        prediction = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        final_pred = polish_turkish(prediction)
        
        score = chrf.compute(predictions=[final_pred], references=[[sample["output"]]], word_order=2)["score"]
        
        results.append({
            "gloss": sample['input'], "expected": sample["output"], 
            "prediction": final_pred, "latency_sec": round(dur, 4), "chrf": round(score, 2),
            "rag_context": context_str
        })

    with open(os.path.join(OUTPUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[SUCCESS] Avg Latency: {total_latency/len(valid_samples):.4f}s")

if __name__ == "__main__":
    main()