import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import threading
from src.model import EfficientVAD_Pro

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "./data/checkpoints/efficient_vad.pth"
DEFAULT_THRESHOLD = 0.107776  # Your optimal value
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# MODEL UTILS
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

# ==========================================
# GUI APPLICATION
# ==========================================
class AnomalyDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Efficient-VAD Pro | Anomaly Detection System")
        self.root.geometry("600x450")
        self.root.resizable(False, False)
        
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = tk.Label(root, text="Video Anomaly Detection", font=("Helvetica", 18, "bold"), bg="#333", fg="white")
        header.pack(fill="x", pady=0)
        
        # Container
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        # 1. Select Video
        tk.Label(frame, text="1. Select Input Video:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.file_frame = tk.Frame(frame)
        self.file_frame.pack(fill="x", pady=5)
        
        self.path_var = tk.StringVar()
        self.entry_path = tk.Entry(self.file_frame, textvariable=self.path_var, width=50)
        self.entry_path.pack(side="left", padx=5)
        
        btn_browse = tk.Button(self.file_frame, text="Browse", command=self.browse_file, bg="#ddd")
        btn_browse.pack(side="left")

        # 2. Threshold
        tk.Label(frame, text="2. Sensitivity Threshold:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 0))
        
        self.thresh_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        self.entry_thresh = tk.Entry(frame, textvariable=self.thresh_var, width=20)
        self.entry_thresh.pack(anchor="w", pady=5)
        tk.Label(frame, text="(Lower = More Sensitive, Higher = Less Sensitive)", font=("Arial", 8, "italic"), fg="gray").pack(anchor="w")

        # 3. Process Button
        self.btn_process = tk.Button(frame, text="START PROCESSING", command=self.start_processing_thread, 
                                     bg="#007bff", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_process.pack(fill="x", pady=30)

        # 4. Progress
        self.progress = ttk.Progressbar(frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x")
        
        self.status_lbl = tk.Label(frame, text="Ready", fg="gray")
        self.status_lbl.pack(pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if filename:
            self.path_var.set(filename)

    def start_processing_thread(self):
        input_path = self.path_var.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
            
        self.btn_process.config(state="disabled")
        self.status_lbl.config(text="Loading Models...", fg="blue")
        
        # Run heavy task in separate thread to keep GUI responsive
        thread = threading.Thread(target=self.process_video, args=(input_path, self.thresh_var.get()))
        thread.start()

    def process_video(self, input_path, threshold):
        try:
            output_path = "Video_result.avi"
            
            # Load Models
            extractor = MobileNetExtractor().to(DEVICE)
            model = EfficientVAD_Pro(input_dim=960).to(DEVICE)
            
            if not os.path.exists(MODEL_PATH):
                self.root.after(0, lambda: messagebox.showerror("Error", "Model checkpoint not found!"))
                self.reset_ui()
                return
                
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            
            transform = get_transform()
            cap = cv2.VideoCapture(input_path)
            
            # Video Setup
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            buffer = []
            scores = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                display_frame = frame.copy()
                
                # Processing Logic (Every 3rd frame)
                if frame_idx % 3 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)
                    feat = extractor(input_tensor)
                    buffer.append(feat)
                    
                    if len(buffer) > 10: buffer.pop(0)
                    
                    current_score = 0.0
                    if len(buffer) == 10:
                        seq = torch.stack(buffer).view(1, 10, 960)
                        with torch.no_grad():
                            recon, _ = model(seq)
                            loss = torch.mean((seq - recon) ** 2).item()
                            current_score = loss
                    scores.append(current_score)
                else:
                    current_score = scores[-1] if scores else 0.0
                
                # Overlay
                color = (0, 255, 0) # Green
                status_text = "NORMAL"
                
                if current_score > threshold:
                    color = (0, 0, 255) # Red
                    status_text = "ANOMALY"
                    
                cv2.putText(display_frame, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display_frame, f"Score: {current_score:.4f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Bar
                bar_len = int((current_score / (threshold * 3)) * 300)
                bar_len = min(bar_len, 300)
                cv2.rectangle(display_frame, (30, 110), (30 + bar_len, 130), color, -1)
                cv2.rectangle(display_frame, (30, 110), (330, 130), (255,255,255), 2)
                
                out.write(display_frame)
                frame_idx += 1
                
                # Update Progress Bar
                if frame_idx % 10 == 0:
                    progress_val = (frame_idx / total_frames) * 100
                    self.root.after(0, lambda v=progress_val: self.update_progress(v))

            cap.release()
            out.release()
            
            self.root.after(0, lambda: self.finish_processing(output_path))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.reset_ui()

    def update_progress(self, val):
        self.progress['value'] = val
        self.status_lbl.config(text=f"Processing... {int(val)}%")

    def finish_processing(self, filepath):
        self.reset_ui()
        self.status_lbl.config(text="Done!", fg="green")
        self.progress['value'] = 100
        
        choice = messagebox.askyesno("Success", "Video processed successfully!\nDo you want to play it now?")
        if choice:
            os.startfile(filepath) # Windows only command to open file

    def reset_ui(self):
        self.btn_process.config(state="normal")
        self.status_lbl.config(text="Ready", fg="gray")
        self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectorApp(root)
    root.mainloop()