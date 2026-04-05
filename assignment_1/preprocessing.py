import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Preprocessing():
    def __init__(self, num_hidden_neurons):
        self.scaler = StandardScaler()
        self.num_hidden_neurons = num_hidden_neurons
        self.kmeans = KMeans(n_clusters=self.num_hidden_neurons, random_state=42)
        
    def preprocess_data(self, X_all, y_all, X_test_submit):
        
        # Scale data
        self.X_all_scaled = self.scaler.fit_transform(X_all)
        self.X_test_scaled = self.scaler.transform(X_test_submit)
        
        # Split Train/Val
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_all_scaled, y_all, test_size=0.2, random_state=42)
        
        train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).float())
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)
        
        self.centers_init = self.determine_centers()
        
        return self.train_loader, self.val_loader, self.centers_init, self.X_test_scaled
        
    def determine_centers(self):
        
        print("Running K-Means to initialize RBF centers...")
        kmeans = KMeans(n_clusters=self.num_hidden_neurons, random_state=42)
        kmeans.fit(self.X_train)
        
        return kmeans.cluster_centers_