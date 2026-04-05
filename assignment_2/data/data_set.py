import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
import pandas as pd

MODEL_NAME = "ProsusAI/finbert" 
MAX_LEN = 128
BATCH_SIZE = 16


class AspectSentimentDataset(Dataset):
    """
    PyTorch Dataset for Aspect-Based Sentiment Analysis.
    Formats inputs to force the model to look at the sentence in the context of a specific stock target.
    """
    def __init__(self, hf_dataset, tokenizer, max_len):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 3-class sentiment: 0=Negative, 1=Neutral, 2=Positive
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Extract strings and the pre-computed integer label
        target = str(item['target'])
        sentence = str(item['sentence'])
        label = int(item['label'])
        
        # 2. Apply the SOTA Cross-Encoder Formatting
        combined_text = f"Target: {target} [SEP] {sentence}"
        
        # 3. Tokenize
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    @staticmethod
    def create_data_loaders():
        print("Loading tokenizer and synthetic dataset...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 1. Load CSV directly via Hugging Face (Bypasses Pandas and import collisions!)
        dataset = load_dataset("csv", data_files="assignment_2/data/generated_datasets/synthetic_absa_dataset.csv", split="train")
        
        # 2. Filter out any empty rows or LLM formatting hallucinations
        valid_sentiments = ["Negative", "Neutral", "Positive"]
        dataset = dataset.filter(lambda x: 
            x['target'] is not None and 
            x['sentence'] is not None and 
            x['sentiment'] in valid_sentiments
        )
        
        # 3. Convert string sentiments back to integers for PyTorch
        label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        def encode_labels(example):
            example['label'] = label_map[example['sentiment']]
            return example
            
        dataset = dataset.map(encode_labels)
        
        # 4. Split 80/20 for Train/Validation
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_data = split_dataset['train']
        val_data = split_dataset['test']

        print(f"Cleaned Train size: {len(train_data)} | Validation size: {len(val_data)}")

        # 5. Pass into your custom PyTorch Dataset (Keep this exactly as you had it!)
        # (Assuming your custom class is named AspectSentimentDataset or FinancialNewsDataset)
        train_dataset = AspectSentimentDataset(train_data, tokenizer, MAX_LEN)
        val_dataset = AspectSentimentDataset(val_data, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, tokenizer


if __name__ == "__main__":

    # Sanity Test
    train_loader, val_loader, tokenizer = AspectSentimentDataset.create_data_loaders()
    
    batch = next(iter(train_loader))
    print("\nSample Batch Shapes:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Attention Mask: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['labels'].shape}")