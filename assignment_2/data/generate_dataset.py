import torch
import json
import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import warnings

def load_slm_annotator():
    """Loads the Qwen 2.5 SLM natively in 16-bit precision."""
    print("Loading Qwen 2.5 1.5B Annotator...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda", 
        torch_dtype=torch.float16 
    )
    
    annotator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50, 
        max_length = None,      
        return_full_text=False,  
        do_sample=False,         
        temperature=0.0          
    )
    
    return annotator, tokenizer

def extract_json_from_response(response_text):
    """Utility to safely extract a JSON object."""
    try:
        match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return None

def build_synthetic_dataset(num_samples=100):
    print("Loading raw financial text...")
    raw_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    
    annotator, tokenizer = load_slm_annotator()
    synthetic_data = []
    
    print(f"\n--- Starting Knowledge Distillation ({num_samples} samples) ---")
    
    for i in tqdm(range(min(num_samples, len(raw_dataset)))):
        text = raw_dataset[i]['text']
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert financial NLP system. Your task is Aspect-Based Sentiment Analysis. "
                    "RULES: "
                    "1. Extract the primary stock ticker or company being evaluated. "
                    "2. Do NOT extract the names of analyst firms, banks, or research groups (e.g., JPMorgan, Nomura, Credit Suisse, Oppenheimer, Piper Jaffray). "
                    "3. If a headline says '[Bank] cuts [Company]', the target is [Company]. "
                    "4. Words like 'cuts', 'weakness', 'downgrade', 'reels in', or 'slides' indicate Negative sentiment. "
                    "Respond ONLY with a valid JSON object: {\"target\": \"Entity Name\", \"sentiment\": \"Positive\" | \"Neutral\" | \"Negative\"}. "
                    "No markdown, no preamble."
                )
            },
            {"role": "user", "content": f"Text: \"{text}\""}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = annotator(prompt)[0]['generated_text']
        parsed_data = extract_json_from_response(output)
        
        if parsed_data and "target" in parsed_data and "sentiment" in parsed_data:
            synthetic_data.append({
                "sentence": text,
                "target": str(parsed_data["target"]),
                "sentiment": str(parsed_data["sentiment"])
            })
            
    print(f"\nSuccessfully generated {len(synthetic_data)} clean samples out of {num_samples} attempts.")
    df = pd.DataFrame(synthetic_data)
    df.to_csv("datasets/generated_datasets/synthetic_absa_dataset.csv", index=False)
    print("Saved to synthetic_absa_dataset.csv!")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    total_rows = len(dataset)

    build_synthetic_dataset(num_samples=total_rows)