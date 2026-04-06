import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import pandas as pd
from models import BERTLoRAModel

def load_hf_model(model_name, device):
    """Loads a standard HuggingFace Sequence Classification model."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True
    ).to(device)
    
    model.eval()
    return model, tokenizer

def load_custom_model(checkpoint_path, base_model_name, device):
    """Loads your custom LoRA fine-tuned model."""
    print("Loading Custom Model from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = BERTLoRAModel.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    return model, tokenizer

def predict(sentence, model, tokenizer, device, label_map, is_custom=False):
    """Generic inference function handling both HF and custom Lightning models."""
    inputs = tokenizer(
        text=sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        if is_custom:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
    predicted_class_id = torch.argmax(probabilities).item()
    
    prediction = label_map[predicted_class_id]
    confidence = probabilities[predicted_class_id].item() * 100
    
    return f"{prediction} ({confidence:.1f}%)"

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization started on {device} ---\n")
    
    # Baseline: Twitter RoBERTa (General social media sentiment)
    # RoBERTa labels are 0:Negative, 1:Neutral, 2:Positive. Mapped to crypto terms.
    baseline_model, baseline_tok = load_hf_model("cardiffnlp/twitter-roberta-base-sentiment", device)
    baseline_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
    
    # CryptoBERT
    cryptobert_model, cryptobert_tok = load_hf_model("ElKulako/cryptobert", device)
    cryptobert_map = {int(k): v.upper() for k, v in cryptobert_model.config.id2label.items()}
    
    # Custom Model 1.5B Qwen 2.5 Distilled 
    CHECKPOINT_PATH = "assignment_2/checkpoints/TweetBERT-lora-epoch=4-qwen-1.5b.ckpt"
    custom_model, custom_tok = load_custom_model(CHECKPOINT_PATH, "cardiffnlp/twitter-roberta-base", device)
    custom_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

    print("\nAll models successfully loaded!\n")

    test_cases = [
        "buying the dip on ETH right now, it's about to explode 🚀",
        "the market is bleeding, cut your losses and sell now",
        "hello i'm good nigerian prince, please send me all of your btc",
        "thanks admin, i will check my dm", 
        "congrats and fuck you. true diamond hands 💎 ✋ if this is true.", 
        "chipotle is now accepting shib as a form of payment.",
        "cheapest i can find in uae atm is 489k aed.",
        "paper handed bitch unless proven otherwise",
        "fuck!! you've been an absolute monster in the market. You've smashed it!!",
        "Fuck you man!! I seriously can't believe you got that stock for so cheap." 
    ]

    test_cases.extend([
        # Extreme Praise/Jealousy
        "You disgust me. Enjoy the early retirement, asshole.", # Expected: BULLISH
        "Holy shit, my wife's boyfriend is going to be so happy with these gains.", # Expected: BULLISH
        "Disgusting. Unbelievable. You held through an 80% drawdown and actually made it out rich. Fuck you.", # Expected: BULLISH
        
        # Sarcasm & "Copium"
        "Wow, another 20% drop. My portfolio is doing absolutely fantastic today.", # Expected: BEARISH
        "This is fine. We are just consolidating before the next massive leg up, right guys? Right?", # Expected: BEARISH (or Neutral)
        "Oh no, Bitcoin crashed to $60k, whatever will I do? *aggressively buys more*", # Expected: BULLISH
        
        # Pure Panic & Capitulation 
        "I'm officially tapped out. Margin called and liquidated. See you guys at McDonald's.", # Expected: BEARISH
        "Catching a falling knife right now. Blood in the streets and I have no fiat left.", # Expected: BEARISH
        "Another massive rug pull. Devs dumped their bags and locked the telegram.", # Expected: BEARISH
        
        # Technical Analysis Jargon 
        "BTC just broke through the 200 SMA on the 4H chart. Golden cross incoming.", # Expected: BULLISH
        "Looking at a textbook head and shoulders pattern here. Breakdown below support confirms it.", # Expected: BEARISH
        "Revenue is up but forward guidance is absolute trash. Puts are going to print at open.", # Expected: BEARISH
        
        # Scams, Phishing & Bot Noise 
        "Click the link in my bio to double your Doge instantly! Limited time airdrop!", # Expected: BEARISH (or Noise/Neutral depending on your mapping)
        "DM me 'CRYPTO' and I'll add you to my VIP signals group where we print money daily 🤑", # Expected: BEARISH
        
        # Irrelevant Meta-Chatter & News 
        "Why was my post deleted? Mods are literally acting like dictators today.", # Expected: NEUTRAL
        "Anyone know what time the FOMC meeting starts in EST?", # Expected: NEUTRAL
        "Honestly I forgot I even had a Coinbase account until I saw this thread.", # Expected: NEUTRAL
        "Can someone link the discord server? The old invite expired.", # Expected: NEUTRAL
        "Solana network is currently experiencing an outage, validators are working on a restart.", # Expected: NEUTRAL
        
        # The "Diamond Hands" Extremists 
        "They want you to panic sell. Not giving up a single satoshi. Riding this to zero or Valhalla." # Expected: BULLISH
    ])
    
    print("="*80)
    print(" RUNNING COMPARATIVE INFERENCE TESTS ")
    print("="*80 + "\n")
    
    results = []
    
    for sentence in test_cases:

        base_pred = predict(sentence, baseline_model, baseline_tok, device, baseline_map, is_custom=False)
        crypto_pred = predict(sentence, cryptobert_model, cryptobert_tok, device, cryptobert_map, is_custom=False)
        custom_pred = predict(sentence, custom_model, custom_tok, device, custom_map, is_custom=True)
        
        results.append({
            "Tweet": sentence,
            "Baseline (RoBERTa)": base_pred,
            "CryptoBERT": crypto_pred,
            "Custom (Ours)": custom_pred
        })

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 200)
    df_results = pd.DataFrame(results)
    
    print(df_results.to_string(index=False))