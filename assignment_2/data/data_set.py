import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base" 
MAX_LEN = 128
BATCH_SIZE = 16

class CryptoSentimentDataset(Dataset):
    """
    PyTorch Dataset for Global Crypto Twitter Sentiment Analysis.
    Formats raw tweets for RoBERTa Sequence Classification.
    """
    def __init__(self, hf_dataset, tokenizer, max_len):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        sentence = str(item['sentence'])
        label = int(item['label'])
                
        encoding = self.tokenizer(
            text=sentence, 
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
        print(f"Loading tokenizer ({MODEL_NAME}) and synthetic dataset...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        dataset = load_dataset("csv", data_files="assignment_2/data/generated_datasets/crypto_distilled_dataset.csv", split="train")
        
        valid_labels = ["Bearish", "Neutral", "Bullish"]
        dataset = dataset.filter(lambda x: 
            x['sentence'] is not None and 
            x['label'] in valid_labels
        )
        
        # Convert string labels to integers for PyTorch
        label_map = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
        def encode_labels(example):
            example['label'] = label_map[example['label']]
            return example
            
        dataset = dataset.map(encode_labels)
        
        # Split 80/20 for Train/Validation
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_data = split_dataset['train']
        val_data = split_dataset['test']

        print(f"Cleaned Train size: {len(train_data)} | Validation size: {len(val_data)}")

        train_dataset = CryptoSentimentDataset(train_data, tokenizer, MAX_LEN)
        val_dataset = CryptoSentimentDataset(val_data, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    
    # Sanity Test
    train_loader, val_loader, tokenizer = CryptoSentimentDataset.create_data_loaders()
    
    batch = next(iter(train_loader))
    print("\nSample Batch Shapes:")
    print(f"Input IDs: {batch['input_ids'].shape}")
    print(f"Attention Mask: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['labels'].shape}")