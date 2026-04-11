import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import warnings

from data import CryptoSentimentDataset
from models import BERTLoRAModel

def main():
    warnings.filterwarnings("ignore")

    print("--- Initializing Data Loaders ---")
    train_loader, val_loader, tokenizer = CryptoSentimentDataset.create_data_loaders()

    print("\n--- Initializing Model ---")

    model = BERTLoRAModel(
        model_name="cardiffnlp/twitter-roberta-base",
        num_labels=3,
        learning_rate=2e-4
    )

    print("\n--- Setting up Callbacks and Logger ---")
    checkpoint_callback = ModelCheckpoint(
        dirpath="assignment_2/checkpoints",
        filename="TweetBERT-lora-{epoch:02d}-{val_f1:.4f}",
        save_top_k=1,         
        monitor="val_f1",     
        mode="max"            
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=2,           # Stop if F1 doesn't improve for 2 consecutive epochs
        mode="max",
        verbose=True
    )

    logger = CSVLogger("logs", name="finbert_absa")

    print("\n--- Initializing PyTorch Lightning Trainer ---")
    trainer = pl.Trainer(
        max_epochs=4,                                          
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",                                    
        devices=1,
        log_every_n_steps=10
    )

    print("\n--- Starting Fine-Tuning ---")
    trainer.fit(model, train_loader, val_loader)

    print("\n--- Training Complete! ---")
    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()