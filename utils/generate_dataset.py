import torch
import json
import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import warnings

def load_slm_annotator():
    print("Loading Qwen 2.5 1.5B Annotator onto GPU...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Utilizing Flash Attention and bf16/fp16 for speedups
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda", 
        dtype=torch.float16 
    )
    annotator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150, 
        max_length = None,     
        return_full_text=False,  
        do_sample=False,         
        temperature=0.0          
    )
    return annotator, tokenizer

def extract_json_from_response(response_text):
    try:
        match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return None

def build_distilled_crypto_dataset(target_size=10000):
    print("Downloading the unlabelled CryptoBERT dataset...")
    raw_dataset = load_dataset("ElKulako/cryptobert-posttrain", split="train")
    
    raw_dataset = raw_dataset.shuffle(seed=42)
    
    annotator, tokenizer = load_slm_annotator()
    synthetic_data = []
    
    print(f"\nBeginning Knowledge Distillation. Target Size: {target_size} rows.")
    
    pbar = tqdm(total=target_size, desc="Generating Dataset")
    
    for i in range(len(raw_dataset)):

        # Stop if target size is hit
        if len(synthetic_data) >= target_size:
            break
            
        raw_text = str(raw_dataset[i]['text'])
        
        # Strip URLs and Links
        clean_text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)
        
        # Strip extra spaces left behind by the link removal
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Length Filter 
        words = clean_text.split()
        if len(words) < 6 or len(words) > 60: 
            continue
            
        # Spam & Bot Filter
        spam_keywords = ["airdrop", "giveaway", "guar free", "buying tokens today", "subscribe"]
        if any(spam in clean_text.lower() for spam in spam_keywords):
            continue
            
        # COT Prompt
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a strict data annotator building a cryptocurrency dataset. "
                    "You must evaluate the text using this exact logical matrix:\n"
                    "- IF the text is irrelevant conversational chatter -> You MUST label as 'Noise'.\n"
                    "- IF the text is a scam, phishing attempt (e.g., 'send me BTC', 'guaranteed returns'), or bot spam -> You MUST label as 'Bearish'.\n"
                    "- IF the text expresses optimism, buying, holding (HODL), or upward price movement overall, despite having profanity -> You MUST label as 'Bullish'.\n"
                    "- IF the text expresses fear, selling, cutting losses, or uses negative slang (e.g., 'paper hands', 'crashing') -> You MUST label as 'Bearish'.\n\n"
                    "Step 1: Write a brief reasoning that strictly identifies which of the IF conditions applies.\n"
                    "Step 2: Output the exact label dictated by the matrix.\n\n"
                    "Respond ONLY with a valid JSON object in this format:\n"
                    "{\"reasoning\": \"This matches the scam condition because...\", \"sentiment\": \"Bullish\" | \"Bearish\" | \"Neutral\" | \"Noise\"}"
                )
            },
            {"role": "user", "content": f"Text: \"{clean_text}\""}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = annotator(prompt)[0]['generated_text']
        parsed_data = extract_json_from_response(output)
        
        if parsed_data and "sentiment" in parsed_data:
            sentiment = str(parsed_data["sentiment"])
            
            # Drop Noise labels
            if sentiment in ["Bullish", "Bearish", "Neutral"]:
                synthetic_data.append({
                    "sentence": clean_text,
                    "label": sentiment
                })
                pbar.update(1) 
                
    pbar.close()
    
    df = pd.DataFrame(synthetic_data)
    
    import os
    os.makedirs("assignment_2/data/generated_datasets", exist_ok=True)
    
    save_path = "assignment_2/data/generated_datasets/crypto_distilled_dataset.csv"
    df.to_csv(save_path, index=False)
    print(f"\nSuccess! Saved {len(df)} cleanly labelled rows to {save_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    TARGET_DATASET_SIZE = 12000 
    build_distilled_crypto_dataset(target_size=TARGET_DATASET_SIZE)