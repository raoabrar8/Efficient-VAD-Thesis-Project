import time
import torch
import numpy as np
from torchvision import models
from src.model import EfficientVAD_Pro

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmark():
    print(f"[INFO] Benchmarking Efficient-VAD Pro on {DEVICE}...")
    print("-" * 40)
    
    # 1. Load Models
    # MobileNetV3 Large
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
    except:
        weights = "pretrained"
    extractor = models.mobilenet_v3_large(weights=weights).to(DEVICE)
    extractor.eval()
    
    # Our Pro Model
    model = EfficientVAD_Pro(input_dim=960).to(DEVICE)
    model.eval()
    
    # 2. Dummy Data (1 Frame)
    dummy_frame = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_seq = torch.randn(1, 10, 960).to(DEVICE)
    
    # 3. Warmup (Get GPU/CPU ready)
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            _ = extractor(dummy_frame)
            _ = model(dummy_seq)
            
    # 4. Test Loop
    num_frames = 500
    print(f"Processing {num_frames} frames...")
    
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_frames):
            # Simulate the full pipeline
            # 1. Feature Extraction
            feat = extractor(dummy_frame)
            # 2. Anomaly Scoring
            _ = model(dummy_seq)
            
    end = time.time()
    total_time = end - start
    
    fps = num_frames / total_time
    
    print("-" * 40)
    print(f"⚡ SPEED RESULTS ⚡")
    print("-" * 40)
    print(f"Total Time : {total_time:.4f} seconds")
    print(f"Raw FPS    : {fps:.2f} FPS")
    
    # Since we skip 2 frames (process every 3rd), the effective monitoring speed is 3x
    effective_fps = fps * 3
    print(f"Real-time  : {effective_fps:.2f} FPS (with Frame Skip=3)")
    print("-" * 40)
    
    if effective_fps > 25:
        print("✅ Result: REAL-TIME CAPABLE")
    else:
        print("⚠️ Result: Near Real-Time")

if __name__ == "__main__":
    benchmark()