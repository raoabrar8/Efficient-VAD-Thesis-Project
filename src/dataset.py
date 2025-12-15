import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class AutoencoderDataset(Dataset):
    """
    Loads feature files for Autoencoder training.
    
    Task: Reconstruction.
    Input: Sequence of 'seq_len' vectors.
    Target: The SAME sequence (Reconstruction).
    """
    def __init__(self, feature_dir, seq_len=10):
        self.feature_dir = feature_dir
        self.seq_len = seq_len
        
        self.file_paths = glob.glob(os.path.join(feature_dir, "*.npy"))
        self.data_cache = [] 
        self.indices = []    
        
        if not self.file_paths:
            print(f"[WARNING] No .npy files found in {feature_dir}")
        
        # Load all data into RAM (Features are small text/binary files, so this is fast)
        for i, path in enumerate(self.file_paths):
            try:
                data = np.load(path) # Shape: (Total_Frames, 2048)
                
                # We need at least seq_len frames to create one sample
                if data.shape[0] < seq_len:
                    continue
                
                self.data_cache.append(data)
                
                # Calculate how many sequences we can make
                # If Total=100, Seq=10 -> Last start index = 90
                last_start_idx = data.shape[0] - seq_len
                
                cache_idx = len(self.data_cache) - 1
                for start_idx in range(last_start_idx + 1):
                    self.indices.append((cache_idx, start_idx))
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
                
        print(f"[INFO] Dataset Ready. Total sequences: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cache_idx, start_idx = self.indices[idx]
        data = self.data_cache[cache_idx]
        
        # Get the window
        window = data[start_idx : start_idx + self.seq_len]
        
        # Convert to Tensor
        seq_tensor = torch.from_numpy(window).float() 
        
        # For Autoencoder: Input == Target
        return seq_tensor, seq_tensor