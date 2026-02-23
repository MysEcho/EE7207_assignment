import os

import numpy as np
import scipy.io


def load_mat_file(path, key_hints):

    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    
    mat = scipy.io.loadmat(path)

    for key in mat.keys():
        if not key.startswith('__'):
            # Check if it matches any hint or just return the first valid data
            if any(hint in key for hint in key_hints):
                return mat[key]
            
    # Return the first non-meta key found
    for key in mat.keys():
        if not key.startswith('__'):
            return mat[key]
    return None

def analyze_dataset():
    print("="*60)
    print(" DATASET INSPECTION REPORT")
    print("="*60)

    print(f"[{'LOADING':^10}] Reading .mat files...")
    X_train = load_mat_file('assignment/dataset/data_train.mat', ['data', 'train', 'X'])
    y_train = load_mat_file('assignment/dataset/label_train.mat', ['label', 'y'])
    X_test  = load_mat_file('assignment/dataset/data_test.mat', ['data', 'test'])

    if X_train is None or y_train is None:
        print("\n CRITICAL: Could not load training data. Aborting.")
        return

    print(f"\n[{'SHAPES':^10}] Checking dimensionality...")
    print(f"  • Train Data:   {X_train.shape}")
    print(f"  • Train Labels: {y_train.shape}")
    print(f"  • Test Data:    {X_test.shape if X_test is not None else 'None'}")

    if X_train.shape[0] != y_train.shape[0]:
        print(f"MISMATCH: {X_train.shape[0]} samples vs {y_train.shape[0]} labels!")
    else:
        print("Sample counts match.")

    print(f"\n[{'LABELS':^10}] Inspecting label values...")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(f"Unique Values: {unique_labels}")
    print(f"Class Counts:  {dict(zip(unique_labels, counts))}")
    
    if -1 in unique_labels:
        print(" WARNING: Found label '-1'. PyTorch BCE Loss requires labels {0, 1}.")
    
    imbalance_ratio = max(counts) / min(counts)
    if imbalance_ratio > 10:
        print(f"WARNING: Severe class imbalance (Ratio 1:{imbalance_ratio:.1f}).")

    print(f"\n[{'FEATURES':^10}] Checking feature statistics...")
    
    # NaNs/Infs
    if np.isnan(X_train).any():
        print("CRITICAL: Train data contains NaNs!")
    if np.isinf(X_train).any():
        print("CRITICAL: Train data contains Infs!")
        
    min_val, max_val = X_train.min(), X_train.max()
    mean_val, std_val = X_train.mean(), X_train.std()
    
    print(f"Global Min:  {min_val:.4f}")
    print(f"Global Max:  {max_val:.4f}")
    print(f"Global Mean: {mean_val:.4f}")
    print(f"Global Std:  {std_val:.4f}")

    if abs(max_val - min_val) > 100 or abs(mean_val) > 10:
        print("WARNING: Data is not normalized. RBF networks use Euclidean distance.")

    if X_test is not None:
        print(f"\n[{'SHIFT':^10}] Comparing Train vs Test statistics...")
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        
        dist_shift = np.linalg.norm(train_mean - test_mean)
        print(f"Distance between Train-Mean and Test-Mean: {dist_shift:.4f}")
        
        if dist_shift > 1.0: 
            print("WARNING: Possible covariate shift. Test data looks different from Train data.")

    print("\n" + "="*60)
    print(" END REPORT")
    print("="*60)

if __name__ == "__main__":
    analyze_dataset()