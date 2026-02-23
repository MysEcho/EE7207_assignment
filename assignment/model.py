import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class RBFGaussianLayer(nn.Module):
    """
    Radial Basis Function Implementation.
    f(x) = exp(-beta * ||x - center||^2)
    """
    def __init__(self, in_features, num_centers, init_centers=None):
        super().__init__()
        self.in_features = in_features
        self.num_centers = num_centers

        # Centers: Shape (num_centers, in_features)
        if init_centers is not None:
            # Ensure tensor form
            if isinstance(init_centers, np.ndarray):
                init_centers = torch.from_numpy(init_centers).float()
            self.centers = nn.Parameter(init_centers)
        else:
            self.centers = nn.Parameter(torch.randn(num_centers, in_features))

        # Beta (Inverse width): Shape (num_centers)
        # Trainable Beta initialized to 1.0 (sigma = 1/sqrt(2*beta) = 1)
        self.beta = nn.Parameter(torch.ones(num_centers) * 1.0)

    def forward(self, x):
        """
        Input x: (batch_size, in_features)
        Output: (batch_size, num_centers)
        """
        batch_size = x.size(0)

        # Compute pairwise distance between X and Centers
        # x shape: (batch_size, 1, in_features)
        # centers shape: (1, num_centers, in_features)
        x_expanded = x.unsqueeze(1) 
        c_expanded = self.centers.unsqueeze(0)

        # Squared Euclidean distance
        dist_sq = torch.sum((x_expanded - c_expanded) ** 2, dim=2)
        
        # Apply Gaussian activation
        # dist_sq (batch_size, num_centers)
        # beta (num_centers)
        return torch.exp(-self.beta * dist_sq)

class RBFNetwork(pl.LightningModule):
    def __init__(self, in_features, num_centers, init_centers=None, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters(ignore=['init_centers']) # Save params for checkpoints
        self.lr = learning_rate

        # Input Layer: 'in_features'
        # Hidden Layer: 'num_centers' neurons (Gaussian RBF)
        self.rbf_layer = RBFGaussianLayer(in_features, num_centers, init_centers)
        
        self.output_layer = nn.Linear(num_centers, 1)
        
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):

        rbf_out = self.rbf_layer(x)

        logits = self.output_layer(rbf_out)
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds.float() == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds.float() == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)