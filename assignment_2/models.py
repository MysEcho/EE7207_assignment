import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics import Accuracy, F1Score

class FinBERTLoRAModel(pl.LightningModule):
    def __init__(self, model_name="ProsusAI/finbert", num_labels=3, learning_rate=2e-4):
        super().__init__()

        self.save_hyperparameters()
        
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True 
        )
        
        # Configure LoRA adapters
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, # Sequence Classification
            r=8,                        # Rank of the update matrices (lower = faster, less memory)
            lora_alpha=16,              # Scaling factor
            lora_dropout=0.1,           # Dropout probability for LoRA layers
            bias="none"                 # Train only the LoRA weights, no biases
        )
        
        # Wrap model with LoRA adapters
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters() 

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def forward(self, input_ids, attention_mask):

        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer