import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
    
    # Return both the raw label (for metrics) and the confidence score (for the dataframe)
    return prediction, confidence

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization started on {device} ---\n")
    
    # Baseline: BERTTweet
    baseline_model, baseline_tok = load_hf_model("cardiffnlp/twitter-roberta-base-sentiment", device)
    baseline_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
    
    # CryptoBERT
    cryptobert_model, cryptobert_tok = load_hf_model("ElKulako/cryptobert", device)
    cryptobert_map = {int(k): v.upper() for k, v in cryptobert_model.config.id2label.items()}
    
    # Custom Model: 1.5B Distilled
    print("\nLoading Custom Model (1.5B Distilled)...")
    CKPT_1_5B = "assignment_2/checkpoints/TweetBERT-lora-epoch=4-qwen-1.5b.ckpt"
    model_1_5b, tok_1_5b = load_custom_model(CKPT_1_5B, "cardiffnlp/twitter-roberta-base", device)
    
    # Custom Model: 7B Distilled
    print("Loading Custom Model (7B Distilled)...")
    CKPT_7B = "assignment_2/checkpoints/TweetBERT-lora-qwen7B-epoch=01-val_f1=0.7536.ckpt"
    model_7b, tok_7b = load_custom_model(CKPT_7B, "cardiffnlp/twitter-roberta-base", device)
    
    custom_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

    print("\nAll 4 models successfully loaded!\n")

    test_cases = [

        {"tweet": "buying the dip on ETH right now, it's about to explode 🚀", "label": "BULLISH"},
        {"tweet": "the market is bleeding, cut your losses and sell now", "label": "BEARISH"},
        {"tweet": "hello i'm good nigerian prince, please send me all of your btc", "label": "BEARISH"},
        {"tweet": "thanks admin, i will check my dm", "label": "NEUTRAL"},
        {"tweet": "congrats and fuck you. true diamond hands 💎 ✋ if this is true.", "label": "BULLISH"},
        {"tweet": "chipotle is now accepting shib as a form of payment.", "label": "BULLISH"},
        {"tweet": "cheapest i can find in uae atm is 489k aed.", "label": "NEUTRAL"},
        {"tweet": "paper handed bitch unless proven otherwise", "label": "BEARISH"},
        {"tweet": "fuck!! you've been an absolute monster in the market. You've smashed it!!", "label": "BULLISH"},
        {"tweet": "Fuck you man!! I seriously can't believe you got that stock for so cheap.", "label": "BULLISH"},

        # Extreme Praise/Jealousy
        {"tweet": "You disgust me. Enjoy the early retirement, asshole.", "label": "BULLISH"},
        {"tweet": "Holy shit, my wife's boyfriend is going to be so happy with these gains.", "label": "BULLISH"},
        {"tweet": "Disgusting. Unbelievable. You held through an 80% drawdown and actually made it out rich. Fuck you.", "label": "BULLISH"},
        
        # Sarcasm & "Copium"
        {"tweet": "Wow, another 20% drop. My portfolio is doing absolutely fantastic today.", "label": "BEARISH"},
        {"tweet": "This is fine. We are just consolidating before the next massive leg up, right guys? Right?", "label": "BEARISH"},
        {"tweet": "Oh no, Bitcoin crashed to $60k, whatever will I do? *aggressively buys more*", "label": "BULLISH"},
        
        # Pure Panic & Capitulation 
        {"tweet": "I'm officially tapped out. Margin called and liquidated. See you guys at McDonald's.", "label": "BEARISH"},
        {"tweet": "Catching a falling knife right now. Blood in the streets and I have no fiat left.", "label": "BEARISH"},
        {"tweet": "Another massive rug pull. Devs dumped their bags and locked the telegram.", "label": "BEARISH"},
        
        # Technical Analysis Jargon 
        {"tweet": "BTC just broke through the 200 SMA on the 4H chart. Golden cross incoming.", "label": "BULLISH"},
        {"tweet": "Looking at a textbook head and shoulders pattern here. Breakdown below support confirms it.", "label": "BEARISH"},
        {"tweet": "Revenue is up but forward guidance is absolute trash. Puts are going to print at open.", "label": "BEARISH"},
        
        # Scams, Phishing & Bot Noise 
        {"tweet": "Click the link in my bio to double your Doge instantly! Limited time airdrop!", "label": "BEARISH"},
        {"tweet": "DM me 'CRYPTO' and I'll add you to my VIP signals group where we print money daily 🤑", "label": "BEARISH"},
        
        # Irrelevant Meta-Chatter & News 
        {"tweet": "Why was my post deleted? Mods are literally acting like dictators today.", "label": "NEUTRAL"},
        {"tweet": "Anyone know what time the FOMC meeting starts in EST?", "label": "NEUTRAL"},
        {"tweet": "Honestly I forgot I even had a Coinbase account until I saw this thread.", "label": "NEUTRAL"},
        {"tweet": "Can someone link the discord server? The old invite expired.", "label": "NEUTRAL"},
        {"tweet": "Solana network is currently experiencing an outage, validators are working on a restart.", "label": "NEUTRAL"},
        
        # The "Diamond Hands" Extremists 
        {"tweet": "They want you to panic sell. Not giving up a single satoshi. Riding this to zero or Valhalla.", "label": "BULLISH"}
    ]
    
    print("="*120)
    print(" RUNNING COMPARATIVE INFERENCE TESTS (ABLATION STUDY) ")
    print("="*120 + "\n")
    
    results = []
    
    # Tracking lists for metrics
    y_true = []
    preds_baseline = []
    preds_crypto = []
    preds_1_5b = []
    preds_7b = []
    
    for case in test_cases:
        sentence = case["tweet"]
        human_label = case["label"]
        y_true.append(human_label)

        base_pred, base_conf = predict(sentence, baseline_model, baseline_tok, device, baseline_map, is_custom=False)
        crypto_pred, crypto_conf = predict(sentence, cryptobert_model, cryptobert_tok, device, cryptobert_map, is_custom=False)
        
        pred_1_5b, conf_1_5b = predict(sentence, model_1_5b, tok_1_5b, device, custom_map, is_custom=True)
        pred_7b, conf_7b = predict(sentence, model_7b, tok_7b, device, custom_map, is_custom=True)
        
        preds_baseline.append(base_pred)
        preds_crypto.append(crypto_pred)
        preds_1_5b.append(pred_1_5b)
        preds_7b.append(pred_7b)
        
        results.append({
            "Tweet": sentence,
            "Human": human_label,
            "Baseline": f"{base_pred} ({base_conf:.1f}%)",
            "CryptoBERT": f"{crypto_pred} ({crypto_conf:.1f}%)",
            "Ours (1.5B)": f"{pred_1_5b} ({conf_1_5b:.1f}%)",
            "Ours (7B)": f"{pred_7b} ({conf_7b:.1f}%)"
        })

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 300)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # METRICS & CONFUSION MATRIX EVALUATION 

    print("\n" + "="*120)
    print(" EVALUATION METRICS (Macro Average) ")
    print("="*120 + "\n")
    
    models_dict = {
        "Baseline (Twitter RoBERTa)": preds_baseline,
        "CryptoBERT": preds_crypto,
        "Ours (1.5B)": preds_1_5b,
        "Ours (7B)": preds_7b
    }
    
    labels_order = ["BEARISH", "NEUTRAL", "BULLISH"]
    metrics_data = []
    
    for model_name, y_pred in models_dict.items():
        acc = accuracy_score(y_true, y_pred)
        # Using macro average to treat all classes equally
        prec = precision_score(y_true, y_pred, labels=labels_order, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, labels=labels_order, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels_order, average='macro', zero_division=0)
        
        metrics_data.append({
            "Model": model_name, 
            "Accuracy": f"{acc:.4f}", 
            "Precision": f"{prec:.4f}", 
            "Recall": f"{rec:.4f}", 
            "F1 Score": f"{f1:.4f}"
        })
        
    df_metrics = pd.DataFrame(metrics_data)
    print(df_metrics.to_string(index=False))

    print("\n" + "="*120)
    print(" CONFUSION MATRICES ")
    print(f" Rows = True Label | Columns = Predicted Label | Order = {labels_order}")
    print("="*120 + "\n")

    for model_name, y_pred in models_dict.items():
        print(f"--- {model_name} ---")
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        print(cm_df)
        print("\n")