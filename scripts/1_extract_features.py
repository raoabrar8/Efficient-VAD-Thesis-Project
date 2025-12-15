import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_config

# ==========================================
# CONFIGURATION
# ==========================================
config = get_config()
VIDEO_DIRS = {
    "train": config.paths.train_videos,
    "test": config.paths.test_videos
}
OUTPUT_ROOT = config.paths.features_dir

# SETTINGS
IMG_SIZE = config.feature_extraction.image_size
FRAME_SKIP = config.feature_extraction.frame_skip
BATCH_SIZE = config.feature_extraction.batch_size
DEVICE = config.get_device()

# ==========================================
# MODEL: MOBILENET V3
# ==========================================
class MobileNetExtractor(nn.Module):
    def __init__(self):
        super(MobileNetExtractor, self).__init__()
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        except:
            weights = "pretrained"
        mobilenet = models.mobilenet_v3_large(weights=weights)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.eval() 

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x) 
            x = self.avgpool(x)
            x = x.flatten(1)
        return x

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ==========================================
# MEMORY-SAFE PROCESSOR
# ==========================================
def process_video(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_buffer = []
    features_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % FRAME_SKIP == 0:
            try:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frames_buffer.append(frame)
            except: pass

        # Process batch immediately to save RAM
        if len(frames_buffer) >= BATCH_SIZE:
            batch = torch.stack(frames_buffer).to(DEVICE)
            with torch.no_grad():
                feats = model(batch)
            features_list.append(feats.cpu().numpy())
            frames_buffer = [] # Clear memory

        frame_count += 1
        
    # Process remaining
    if len(frames_buffer) > 0:
        batch = torch.stack(frames_buffer).to(DEVICE)
        with torch.no_grad():
            feats = model(batch)
        features_list.append(feats.cpu().numpy())

    cap.release()
    
    if not features_list:
        return None
    arr = np.concatenate(features_list, axis=0)
    # Shape check: must be 2D and arr.shape[1] == 960
    if arr.ndim != 2 or arr.shape[1] != 960 or arr.shape[0] < 2:
        # Log bad feature
        with open('bad_features_deleted.txt', 'a') as log:
            log.write(f"{video_path} -> shape {arr.shape}\n")
        return None
    return arr

# ==========================================
# MAIN EXECUTION (WITH RESUME LOGIC)
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] 🚀 Starting MobileNet Extraction on {DEVICE}...")
    
    model = MobileNetExtractor().to(DEVICE)
    transform = get_transform()
    
    for split, video_dir in VIDEO_DIRS.items():
        print(f"\n[INFO] Scanning: {video_dir}")
        
        # Find videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(video_dir, "**", ext), recursive=True))
            
        print(f"[INFO] Found {len(video_files)} videos for '{split}'. Checking for existing files...")
        
        # Filter out videos that are already processed
        videos_to_process = []
        for v_path in video_files:
            rel_path = os.path.relpath(v_path, video_dir)
            save_name = os.path.splitext(rel_path)[0] + ".npy"
            save_path = os.path.join(OUTPUT_ROOT, split, save_name)
            
            # RESUME LOGIC:
            if os.path.exists(save_path):
                # check if file is not empty (corruption check)
                if os.path.getsize(save_path) > 0:
                    continue # Skip, it's already done
                else:
                    print(f"[WARN] Found corrupted file for {os.path.basename(v_path)}. Re-processing.")
                    os.remove(save_path)
            
            videos_to_process.append(v_path)
            
        print(f"[INFO] Resuming extraction on {len(videos_to_process)} new videos.")
        
        if len(videos_to_process) == 0:
            continue

        # Process Loop
        for v_path in tqdm(videos_to_process, desc=f"Extracting {split}"):
            rel_path = os.path.relpath(v_path, video_dir)
            save_name = os.path.splitext(rel_path)[0] + ".npy"
            save_path = os.path.join(OUTPUT_ROOT, split, save_name)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            try:
                feats = process_video(v_path, model, transform)
                if feats is not None:
                    np.save(save_path, feats)
            except Exception as e:
                print(f"[ERROR] Failed {v_path}: {e}")
                
    print("\n[SUCCESS] Feature Extraction Complete.")