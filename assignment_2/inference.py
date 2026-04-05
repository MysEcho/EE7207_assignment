import torch
from transformers import AutoTokenizer
import warnings

from models import FinBERTLoRAModel

def load_model_and_tokenizer(checkpoint_path, base_model_name="ProsusAI/finbert"):
    """Loads the tokenizer and the fine-tuned LoRA model from a checkpoint."""
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinBERTLoRAModel.load_from_checkpoint(checkpoint_path).to(device)
    model.eval() 
    
    return model, tokenizer, device

def predict_aspect_sentiment(sentence, target_entity, model, tokenizer, device):
    """Predicts the sentiment of a sentence towards a specific target entity."""
    
    combined_text = f"Target: {target_entity} [SEP] {sentence}"
    
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
    predicted_class_id = torch.argmax(probabilities).item()
    
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    prediction = label_map[predicted_class_id]
    confidence = probabilities[predicted_class_id].item() * 100
    
    return prediction, confidence

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    CHECKPOINT_PATH = "assignment_2/checkpoints/finbert-lora-epoch=03-val_f1=0.7313.ckpt" 
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(CHECKPOINT_PATH)
        print(f"Model successfully loaded onto {device}!\n")
        
        test_cases = [
            # 1 & 2: Contrasting Entities (Testing attention boundary)
            {
                "target": "AMD",
                "sentence": "AMD's server chips are eating into Intel's market share at an unprecedented rate."
            },
            {
                "target": "Intel",
                "sentence": "AMD's server chips are eating into Intel's market share at an unprecedented rate."
            },
            
            # 3 & 4: Sub-Entity vs Parent Entity (Mixed signals in one company)
            {
                "target": "AWS",
                "sentence": "Despite AWS posting record-breaking profits, Amazon's retail division dragged overall earnings down."
            },
            {
                "target": "Amazon",
                "sentence": "Despite AWS posting record-breaking profits, Amazon's retail division dragged overall earnings down."
            },

            # 5: Sarcasm / Irony
            {
                "target": "WeWork",
                "sentence": "Another brilliant quarter for WeWork, managing to burn through a billion dollars of cash while pretending to be a tech company."
            },

            # 6: Complex Negation / Double Negative
            {
                "target": "Tesla",
                "sentence": "It is not entirely inaccurate to say that Tesla's latest production numbers weren't a complete disaster."
            },

            # 7: Expectations vs. Reality (Numbers are positive, sentiment is negative)
            {
                "target": "Microsoft",
                "sentence": "Microsoft's earnings grew by an impressive 15%, but since Wall Street priced in 25%, the stock was brutally punished."
            },

            # 8: Guilt by Association / Macro impacts
            {
                "target": "Coinbase",
                "sentence": "The SEC's aggressive crackdown on Binance sent shockwaves through the crypto market, leaving Coinbase as collateral damage."
            },

            # 9: Praise by Association
            {
                "target": "TSMC",
                "sentence": "Apple's stellar iPhone sales over the holiday season inevitably mean a massive windfall for TSMC."
            },

            # 10: Idioms and Financial Jargon
            {
                "target": "Boeing",
                "sentence": "Retail investors are just catching a falling knife with Boeing after the latest regulatory groundings."
            },

            # 11: Conditional / Hypothetical Risk
            {
                "target": "Netflix",
                "sentence": "If Netflix cannot curb password sharing effectively in Europe, their Q4 margins will face severe headwinds."
            },

            # 12 & 13: Acquisition & Competitor Dynamics
            {
                "target": "Twitter",
                "sentence": "Elon Musk's chaotic restructuring of Twitter has been a surprising boon for Bluesky's user acquisition."
            },
            {
                "target": "Bluesky",
                "sentence": "Elon Musk's chaotic restructuring of Twitter has been a surprising boon for Bluesky's user acquisition."
            },

            # 14: Implicit Negative (No explicitly negative words, but bad news)
            {
                "target": "Credit Suisse",
                "sentence": "The unexpected and sudden departure of the CFO has left Credit Suisse investors asking very tough questions."
            },

            # 15: Nuanced Neutral / Balanced clauses
            {
                "target": "Alphabet",
                "sentence": "Alphabet's decision to acquire the cybersecurity firm aligns with their long-term strategy, though the premium paid was undeniably steep."
            },

            # 16: Macro vs Micro Divergence
            {
                "target": "Snowflake",
                "sentence": "While the broader tech sector rallied heavily on cooling inflation, Snowflake's weak forward guidance left it out in the cold."
            },

            # 17: Backhanded Compliment
            {
                "target": "Exxon",
                "sentence": "Exxon's dividend yield looks incredibly attractive right now, but only because the underlying stock price has cratered."
            },

            # 18: Regulatory vs Financial conflict
            {
                "target": "Google",
                "sentence": "The DOJ antitrust lawsuit against Google remains a dark cloud, yet their core ad revenue is practically bulletproof."
            },

            # 19: Turnaround / Temporal Shift
            {
                "target": "Disney",
                "sentence": "After a disastrous box office year, Disney's surprise restructuring finally gives the bulls a reason to smile."
            },

            # 20: Aggressive Negation
            {
                "target": "Ford",
                "sentence": "Ford didn't just miss their quarterly targets; they fundamentally failed to understand the shift in EV consumer demand."
            }
        ]
        
        print(" Running Inference Tests ")
        for test in test_cases:
            target = test["target"]
            sentence = test["sentence"]
            
            sentiment, confidence = predict_aspect_sentiment(sentence, target, model, tokenizer, device)
            
            print(f"Sentence: '{sentence}'")
            print(f"Target Entity: {target}")
            print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f}%)\n")
            
    except FileNotFoundError:
        print(f"Error: Could not find checkpoint at {CHECKPOINT_PATH}.")
        print("Please ensure you have pasted the exact filename from your checkpoints/ directory.")