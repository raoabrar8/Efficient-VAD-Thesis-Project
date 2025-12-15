import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EfficientVAD_Pro
from src.config import get_config

# ==========================================
# CONFIGURATION
# ==========================================
config = get_config()
FEATURE_DIR = os.path.join(config.paths.features_dir, "train")
CHECKPOINT_DIR = config.paths.checkpoints_dir
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "efficient_vad.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# HYPERPARAMETERS
SEQ_LEN = config.model.sequence_length
BATCH_SIZE = config.training.batch_size
EPOCHS = config.training.epochs
LEARNING_RATE = config.training.learning_rate

# MODEL SETTINGS
INPUT_DIM = config.model.input_dim
HIDDEN_DIM = config.model.hidden_dim
LATENT_DIM = config.model.latent_dim
NUM_MEMORIES = config.model.num_memories

DEVICE = config.get_device()

# ==========================================
# DATASET LOADER
# ==========================================
class VideoFeatureDataset(Dataset):
    def __init__(self, feature_dir, seq_len=10):
        self.files = glob.glob(os.path.join(feature_dir, "*.npy"))
        self.data = []
        self.seq_len = seq_len
        
        print(f"[INFO] Loading data from {len(self.files)} files...")
        
        # Pre-load data into memory (since features are small)
        for f in self.files:
            try:
                feats = np.load(f)
                # Create sliding windows with overlap
                # Stride from config means we take a window every N frames
                stride = config.training.sequence_stride
                if len(feats) > seq_len:
                    for i in range(0, len(feats) - seq_len, stride):
                        self.data.append(feats[i:i+seq_len])
            except:
                pass
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

# ==========================================
# CUSTOM LOSS FUNCTION
# ==========================================

def entropy_loss(att_weights):
    """
    Forces the model to be confident.
    Instead of using 10 memory items a little bit, use 1 memory item A LOT.
    This makes the 'Normal' definition sharper.
    """
    # Add small epsilon to avoid log(0)
    entropy = -torch.mean(torch.sum(att_weights * torch.log(att_weights + 1e-12), dim=1))
    return entropy

# ==========================================
# MEMORY SPARSITY LOSS
# ==========================================
def memory_sparsity_loss(att_weights):
    """
    Encourages the attention weights to be sparse (close to one-hot),
    i.e., only one memory slot is highly activated per input.
    Lower L1 norm means more sparse.
    """
    # L1 norm of attention weights (mean over batch)
    sparsity = torch.mean(torch.sum(torch.abs(att_weights), dim=1))
    return sparsity

# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    print(f"[INFO] Starting 'Efficient-VAD' Training on {DEVICE}")
    
    # 1. Load Data
    dataset = VideoFeatureDataset(FEATURE_DIR, SEQ_LEN)
    if len(dataset) == 0:
        print("[ERROR] No features found! Did you run '1_extract_features.py'?")
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[INFO] Training Samples: {len(dataset)} | Batches: {len(train_loader)}")

    # 2. Init Model
    model = EfficientVAD_Pro(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        latent_dim=LATENT_DIM,
        num_memories=NUM_MEMORIES
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_criterion = nn.MSELoss()

    # 3. Train
    model.train()
    
    for epoch in range(EPOCHS):
        total_mse = 0
        total_ent = 0
        start = time.time()
        
        for batch in train_loader:
            batch = batch.to(DEVICE) # Shape: (Batch, Seq, 960)
            
            optimizer.zero_grad()
            
            # Forward Pass
            reconstruction, att_weights = model(batch)
            

            # Calculate Losses
            l_rec = mse_criterion(reconstruction, batch)
            l_ent = entropy_loss(att_weights)
            l_sparsity = memory_sparsity_loss(att_weights)

            # Combined Loss: Reconstruction + entropy + sparsity
            entropy_weight = config.training.entropy_weight
            sparsity_weight = getattr(config.training, 'sparsity_weight', 0.001)  # Default if not in config
            loss = l_rec + (entropy_weight * l_ent) + (sparsity_weight * l_sparsity)
            
            loss.backward()
            optimizer.step()
            

            total_mse += l_rec.item()
            total_ent += l_ent.item()
            # Optionally, track sparsity loss as well
            
        avg_mse = total_mse / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE Loss: {avg_mse:.6f} | Entropy: {total_ent/len(train_loader):.4f} | Time: {time.time()-start:.1f}s")

    # 4. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SUCCESS] Efficient-VAD Pro Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()