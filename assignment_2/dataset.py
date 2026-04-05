import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

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
        
        target_entity = str(item['target'])
        sentence = str(item['sentence'])
        
        # Target: [Entity] [SEP] [Sentence]
        combined_text = f"Target: {target_entity} [SEP] {sentence}"
        
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Discretize scores to labels
        score = float(item['score'])
        
        if score <= -0.1:
            label = 0  # Negative
        elif score >= 0.1:
            label = 2  # Positive
        else:
            label = 1  # Neutral

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    @staticmethod
    def create_data_loaders():
        print("Loading tokenizer and dataset...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
        
        if 'validation' not in dataset:
            split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
            train_data = split_dataset['train']
            val_data = split_dataset['test']
        else:
            train_data = dataset['train']
            val_data = dataset['validation']

        print(f"Train size: {len(train_data)} | Validation size: {len(val_data)}")

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