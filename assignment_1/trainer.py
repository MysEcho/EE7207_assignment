import os

import numpy as np
import pytorch_lightning as pl
import scipy.io
import torch
from model import RBFNetwork
from preprocessing import Preprocessing
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report


class RBFPipeline:
    def __init__(self, mode="train", num_centers=100, max_epochs=150, learning_rate=0.01, ckpt_name:str='best_checkpoint'):
      
        self.mode = mode.lower()
        self.num_centers = num_centers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
        self.data_dir = 'assignment/dataset'
        self.ckpt_dir = 'assignment/checkpoints'
        self.ckpt_name = ckpt_name if ckpt_name.endswith('.ckpt') else f"{ckpt_name}.ckpt"
        
        self.ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)

    def _load_and_preprocess_data(self):
       
        print("Loading data for preprocessing...")
        
        try:
            train_data = scipy.io.loadmat(f'{self.data_dir}/data_train.mat')
            train_label = scipy.io.loadmat(f'{self.data_dir}/label_train.mat')
            test_data = scipy.io.loadmat(f'{self.data_dir}/data_test.mat')

            X_all = train_data.get('data_train', train_data.get('data'))
            y_all = train_label.get('label_train', train_label.get('label'))
            X_test_submit = test_data.get('data_test', test_data.get('data'))

            # -1 to 0 labels for BCE Loss
            y_all[y_all == -1] = 0
            y_all = y_all.reshape(-1, 1).astype(np.float32)

        except FileNotFoundError:
            
            print("Error: Files not found. Generating Dummy Data...")
            
            X_all = np.random.randn(200, 10).astype(np.float32)
            y_all = np.random.randint(0, 2, (200, 1)).astype(np.float32)
            X_test_submit = np.random.randn(50, 10).astype(np.float32)

        self.input_dim = X_all.shape[1]
        
        print("Preprocessing and running K-Means...")
        preprocessor = Preprocessing(num_hidden_neurons=self.num_centers)
        
        return preprocessor.preprocess_data(X_all, y_all, X_test_submit)

    def train(self):

        print("\n" + "="*40 + "\n TRAINING MODE \n" + "="*40)
        
        train_loader, val_loader, centers_init, _ = self._load_and_preprocess_data()

        model = RBFNetwork(
            in_features=self.input_dim, 
            num_centers=self.num_centers, 
            init_centers=centers_init, 
            learning_rate=self.learning_rate
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_dir,          
            filename=self.ckpt_name.replace(".ckpt", ""), 
            save_top_k=1,                   
            monitor="val_loss",            
            mode="min"                      
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs, 
            accelerator="auto", 
            log_every_n_steps=5, 
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model, train_loader, val_loader)
        print(f"\n Training complete. Best model saved at: {checkpoint_callback.best_model_path}")

    def inference(self):

        print("\n" + "="*40 + "\n INFERENCE MODE \n" + "="*40)
        
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}. Please run train mode first.")

        # Load data for validation metrics
        _, val_loader, _, X_test_scaled = self._load_and_preprocess_data()

        print(f"\nLoading checkpoint: {self.ckpt_path}")
        model = RBFNetwork.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            in_features=self.input_dim,
            num_centers=self.num_centers
        )
        model.eval()
        model.freeze()

        print("\n--- Validation Metrics ---")
        X_val = torch.cat([batch[0] for batch in val_loader]).to(model.device)
        y_val = torch.cat([batch[1] for batch in val_loader]).cpu().numpy().astype(int)

        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (torch.sigmoid(val_logits) > 0.5).int().cpu().numpy()

        print(classification_report(y_val, val_preds, target_names=['Class 0 (-1)', 'Class 1 (1)']))

        print("\n--- Test Set Predictions ---")
        test_tensor = torch.from_numpy(X_test_scaled).float().to(model.device)
        
        with torch.no_grad():
            test_logits = model(test_tensor)
            test_preds = (torch.sigmoid(test_logits) > 0.5).int().cpu().numpy()
            
        print(f"Predictions shape: {test_preds.shape}")
        print("Predictions:\n", test_preds[:50].flatten())
        
        # scipy.io.savemat(f'{self.data_dir}/prediction.mat', {'label': np.where(test_preds == 0, -1, 1)})

    def run(self):

        if self.mode == "train":
            self.train()
        elif self.mode == "inference":
            self.inference()
        else:
            print(f"Invalid mode: {self.mode}. Choose 'train' or 'inference'.")