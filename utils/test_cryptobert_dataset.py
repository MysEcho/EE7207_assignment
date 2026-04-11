from datasets import load_dataset
import random
import warnings

def sample_crypto_dataset(num_samples=15):
    print("Fetching the StockTwits Crypto dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("ElKulako/cryptobert-posttrain", split="train")
    
    print(f"\nTotal rows in dataset: {len(dataset)}")
    print(f"Extracting {num_samples} random samples for analysis:\n")
    print("=" * 80)
    
    # Select random indices to ensure we get a diverse look at the data
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(sample_indices, 1):
        raw_text = dataset[idx]['text']
        print(f"Sample {i} (Index {idx}):")
        print(f"{raw_text}")
        print("-" * 80)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sample_crypto_dataset()