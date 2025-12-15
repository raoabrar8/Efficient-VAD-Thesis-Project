import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
NORMAL_VIDEO_DIR = "./data/normal_videos"
TEST_VIDEO_DIR = "./data/test_videos"
OUTPUT_PLOT = "./results_graphs_v3/tsne_visualization.png"

# Limit how many videos to plot to keep it fast and clean
SAMPLES_PER_CLASS = 100 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# FEATURE EXTRACTOR
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_video_feature(video_path, model, transform):
    """Extracts one single averaged 960-dim vector for the whole video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    # Only take first 50 frames to be fast
    while len(frames) < 50:
        ret, frame = cap.read()
        if not ret: break
        if count % 5 == 0: # Skip frames for speed
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frames.append(frame)
            except: pass
        count += 1
    cap.release()
    
    if not frames: return None
    
    # Process batch
    batch = torch.stack(frames).to(DEVICE)
    with torch.no_grad():
        feats = model(batch) # (N, 960)
        
    # Average all frames to get ONE dot for the plot
    avg_feat = torch.mean(feats, dim=0).cpu().numpy()
    return avg_feat

# ==========================================
# MAIN
# ==========================================
def generate_tsne():
    print("[INFO] Initializing Feature Extractor...")
    model = MobileNetExtractor().to(DEVICE)
    transform = get_transform()
    
    # 1. Collect Video Files
    normal_videos = glob.glob(os.path.join(NORMAL_VIDEO_DIR, "**", "*.mp4"), recursive=True)
    test_videos = glob.glob(os.path.join(TEST_VIDEO_DIR, "**", "*.mp4"), recursive=True)
    
    # Filter out "Normal" videos that might be in the test folder
    abnormal_videos = [v for v in test_videos if "Normal" not in v]

    # Shuffle and Limit
    np.random.shuffle(normal_videos)
    np.random.shuffle(abnormal_videos)
    
    normal_videos = normal_videos[:SAMPLES_PER_CLASS]
    abnormal_videos = abnormal_videos[:SAMPLES_PER_CLASS]
    
    print(f"[INFO] extracting features for {len(normal_videos)} Normal and {len(abnormal_videos)} Abnormal videos...")

    X = []
    y = []
    
    # 2. Extract Normal Features
    for v in tqdm(normal_videos, desc="Normal Videos"):
        feat = extract_video_feature(v, model, transform)
        if feat is not None:
            X.append(feat)
            y.append(0) # 0 = Normal (Green)

    # 3. Extract Abnormal Features
    for v in tqdm(abnormal_videos, desc="Abnormal Videos"):
        feat = extract_video_feature(v, model, transform)
        if feat is not None:
            X.append(feat)
            y.append(1) # 1 = Abnormal (Red)

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Data Shape: {X.shape}") # Should be (N, 960)

    # 4. Run t-SNE
    print("[INFO] Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 5. Plot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], 
                c='green', label='Normal', alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
    
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], 
                c='red', label='Abnormal', alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
    
    plt.title("t-SNE Feature Visualization (Efficient-VAD Pro)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"[SUCCESS] Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    generate_tsne()