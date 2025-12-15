import customtkinter as ctk
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
from torchvision import models, transforms
from src.model import EfficientVAD_Pro 
import threading
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# CONFIGURATION
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# UPDATED PATHS & SETTINGS FOR PRO MODEL
MODEL_PATH = "./data/checkpoints/efficient_vad.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
FRAME_SKIP = 3      # Matches your training
SEQ_LEN = 10        # Matches your training
INPUT_DIM = 960     # MobileNetV3 Output
DEFAULT_THRESH = 0.22 # Best threshold (mean + 1 std) from score distributions

# ==========================================
# 1. FEATURE EXTRACTOR (MobileNetV3)
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
# GUI APPLICATION CLASS
# ==========================================
class AnomalyDetectionApp(ctk.CTk):
    def show_error(self, msg):
        ctk.CTkMessageBox(title="Error", message=msg, icon="cancel")

    def show_about(self):
        ctk.CTkMessageBox(title="About Efficient-VAD GUI", message="Efficient-VAD GUI\nM.Phil Thesis Project\nAuthor: Your Name\n2025\n\nFor anomaly detection in surveillance videos.")

    def show_help(self):
        ctk.CTkMessageBox(title="Help", message="1. Load a video.\n2. Click START DETECTION.\n3. Adjust threshold as needed.\n4. Save results for analysis.\n\nContact: your.email@example.com")
    def __init__(self):

        super().__init__()
        self.title("Efficient-VAD GUI")
        self.geometry("1280x800")
        self.minsize(900, 600)
        # Make all rows/columns expandable
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # --- MENU BAR ---
        self.menu_bar = ctk.CTkFrame(self, height=30, fg_color="#222")
        self.menu_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.menu_bar.grid_columnconfigure(0, weight=1)
        about_btn = ctk.CTkButton(self.menu_bar, text="About", width=80, command=self.show_about, fg_color="#333")
        about_btn.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        help_btn = ctk.CTkButton(self.menu_bar, text="Help", width=80, command=self.show_help, fg_color="#333")
        help_btn.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        # --- LEFT SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=0)
        self.sidebar.grid_rowconfigure(99, weight=1)  # Pushes widgets up
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Efficient-VAD\nGUI", font=ctk.CTkFont(size=28, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Select Video", height=40, font=("Arial", 14), command=self.load_video_file)
        self.btn_load.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        
        self.btn_start = ctk.CTkButton(self.sidebar, text="▶ START DETECTION", height=50, font=("Arial", 14, "bold"), 
                           fg_color="green", hover_color="darkgreen", command=self.start_analysis)
        self.btn_start.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.btn_start.configure(state="disabled")

        # Save Results Button
        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Save Results", height=40, font=("Arial", 13), command=self.save_results)
        self.btn_save.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.btn_save.configure(state="disabled")

        # SENSITIVITY SLIDER
        self.slider_frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b")
        self.slider_frame.grid(row=4, column=0, padx=10, pady=30, sticky="ew")
        
        ctk.CTkLabel(self.slider_frame, text="Sensitivity Threshold", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.threshold_var = ctk.DoubleVar(value=DEFAULT_THRESH) 
        
        self.threshold_label = ctk.CTkLabel(self.slider_frame, text=f"{DEFAULT_THRESH:.6f}", font=("Consolas", 16, "bold"), text_color="cyan")
        self.threshold_label.pack(pady=(0, 5))
        
        # Slider range suited for MSE loss (0.0 to 0.5)
        self.slider = ctk.CTkSlider(self.slider_frame, from_=0.01, to=0.5, number_of_steps=100, 
                                    variable=self.threshold_var, command=self.update_threshold_text)
        self.slider.pack(pady=10, padx=10, fill="x")
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Idle", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=(20, 10))

        # --- MAIN AREA ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- STATUS BAR ---
        self.status_bar = ctk.CTkLabel(self, text="Ready", fg_color="#222", anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        # --- PROGRESS BAR ---
        self.progress = ctk.CTkProgressBar(self.sidebar, width=200)
        self.progress.grid(row=7, column=0, padx=20, pady=10)
        self.progress.set(0)
        self.progress.grid_remove()

        # Results storage
        self.scores = []
        self.main_frame.grid_rowconfigure(0, weight=3) # Video
        self.main_frame.grid_rowconfigure(1, weight=1) # Dashboard
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Display
        self.video_label = ctk.CTkLabel(self.main_frame, text="Load a Video to Begin", fg_color="#000000", corner_radius=10)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Dashboard Area (Alerts + Graph)
        self.dashboard = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a")
        self.dashboard.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.dashboard.grid_columnconfigure(0, weight=1) # Alert Area
        self.dashboard.grid_columnconfigure(1, weight=2) # Graph Area
        self.dashboard.grid_rowconfigure(0, weight=1)
        
        # 1. Alert Panel
        self.alert_panel = ctk.CTkFrame(self.dashboard, fg_color="transparent")
        self.alert_panel.grid(row=0, column=0, sticky="nsew", padx=10)
        
        self.alert_label = ctk.CTkLabel(self.alert_panel, text="READY", font=("Arial", 36, "bold"), text_color="gray")
        self.alert_label.pack(expand=True)
        
        self.score_text = ctk.CTkLabel(self.alert_panel, text="Score: 0.0000", font=("Consolas", 20))
        self.score_text.pack(pady=10)

        # 2. Live Graph Panel
        self.graph_frame = ctk.CTkFrame(self.dashboard, fg_color="transparent")
        self.graph_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.init_graph()

        # State
        self.video_path = None
        self.running = False
        self.feature_extractor = None
        self.ae_model = None
        self.transform = None
        
        # Load models after UI init
        self.after(200, self.load_models)

    def init_graph(self):
        """Initialize Matplotlib graph"""
        self.fig, self.ax = plt.subplots(figsize=(5, 2), dpi=80)
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        
        self.x_data = deque(maxlen=50)
        self.y_data = deque(maxlen=50)
        
        self.line, = self.ax.plot([], [], color='cyan', linewidth=2)
        self.thresh_line = self.ax.axhline(y=DEFAULT_THRESH, color='yellow', linestyle='--', alpha=0.5)
        
        self.ax.set_ylim(0, 0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_graph(self, score):
        self.y_data.append(score)
        self.x_data.append(len(self.y_data))
        
        self.line.set_data(range(len(self.y_data)), self.y_data)
        self.thresh_line.set_ydata([self.threshold_var.get()])
        
        self.ax.set_xlim(0, max(10, len(self.y_data)))
        self.ax.set_ylim(0, max(0.2, max(self.y_data)*1.2))
        
        self.canvas.draw()

    def update_threshold_text(self, value):
        self.threshold_label.configure(text=f"{value:.6f}")

    def load_models(self):
        try:
            self.status_label.configure(text="Loading MobileNetV3...", text_color="orange")
            self.status_bar.configure(text="Loading MobileNetV3...")
            self.update()
            self.feature_extractor = MobileNetExtractor().to(DEVICE)
            self.ae_model = EfficientVAD_Pro(input_dim=INPUT_DIM, hidden_dim=256, latent_dim=64)
            if os.path.exists(MODEL_PATH):
                self.ae_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                self.status_label.configure(text="System Ready", text_color="lightgreen")
                self.status_bar.configure(text="System Ready")
            else:
                self.status_label.configure(text="Checkpoint Missing!", text_color="red")
                self.status_bar.configure(text="Checkpoint Missing!")
                self.btn_load.configure(state="disabled")
                self.show_error("Model checkpoint not found at: " + MODEL_PATH)
                return
            self.ae_model.to(DEVICE)
            self.ae_model.eval()
            self.transform = get_transform()
        except Exception as e:
            self.status_label.configure(text=f"Error Loading Models", text_color="red")
            self.status_bar.configure(text=f"Error Loading Models")
            self.show_error(f"Model Load Error: {e}")

    def load_video_file(self):
        try:
            file_path = ctk.filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
            if file_path:
                self.video_path = file_path
                self.btn_start.configure(state="normal", text="▶ START DETECTION")
                self.btn_save.configure(state="disabled")
                self.video_label.configure(text=f"Loaded: {os.path.basename(file_path)}")
                self.alert_label.configure(text="READY", text_color="gray")
                self.scores.clear()
                self.y_data.clear()
                self.update_graph(0)
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret: self.display_frame(frame)
                cap.release()
        except Exception as e:
            self.show_error(f"Video Load Error: {e}")

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        # Use the current label size for scaling
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 640, 360
        # Maintain aspect ratio
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 10 and new_h > 10:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
        self.video_label.configure(image=imgtk, text="")

    def start_analysis(self):
        if not self.video_path: return
        self.running = True
        self.btn_start.configure(state="disabled", text="PROCESSING...", fg_color="gray")
        self.btn_load.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.status_label.configure(text="Analyzing Video Stream...", text_color="#3498db")
        self.status_bar.configure(text="Analyzing Video Stream...")
        self.progress.set(0)
        self.progress.grid()
        thread = threading.Thread(target=self.process_video_logic)
        thread.daemon = True
        thread.start()

    def process_video_logic(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            feature_buffer = [] 
            self.scores.clear()
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            delay = 1.0 / fps
            display_interval = max(1, int(fps // 15))  # Only update UI ~15 fps
            while cap.isOpened() and self.running:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: break
                if frame_count % display_interval == 0:
                    self.after(0, self.display_frame, frame)
                current_score = 0.0
                if frame_count % FRAME_SKIP == 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            feat = self.feature_extractor(input_tensor) # (1, 960)
                        feature_buffer.append(feat)
                        if len(feature_buffer) > SEQ_LEN: feature_buffer.pop(0)
                        if len(feature_buffer) == SEQ_LEN:
                            seq_tensor = torch.stack(feature_buffer).view(1, SEQ_LEN, INPUT_DIM).to(DEVICE)
                            with torch.no_grad():
                                recon, _ = self.ae_model(seq_tensor)
                                loss = torch.mean((recon - seq_tensor) ** 2).item()
                                current_score = loss
                        self.scores.append(current_score)
                        self.after(0, self.update_dashboard, current_score)
                        self.after(0, self.update_graph, current_score)
                    except Exception as e:
                        self.show_error(f"Processing Error: {e}")
                else:
                    if self.scores: current_score = self.scores[-1]
                frame_count += 1
                # Progress update
                if total_frames > 0:
                    self.after(0, self.progress.set, min(1.0, frame_count/total_frames))
                process_time = time.time() - start_time
                wait_time = max(0.001, delay - process_time)
                time.sleep(wait_time)
            cap.release()
            self.running = False
            self.after(0, self.reset_ui)
        except Exception as e:
            self.show_error(f"Video Processing Error: {e}")
    def save_results(self):
        if not self.scores:
            self.show_error("No results to save. Run detection first.")
            return
        import pandas as pd
        import tkinter.filedialog as fd
        import os
        # Ask for a directory to save all results
        out_dir = fd.askdirectory(title="Select Folder to Save Results")
        if not out_dir:
            return
        # Save CSV
        base_name = os.path.splitext(os.path.basename(getattr(self, 'video_path', 'results')))[0]
        csv_path = os.path.join(out_dir, f"{base_name}_scores.csv")
        pd.DataFrame({"frame": list(range(len(self.scores))), "score": self.scores}).to_csv(csv_path, index=False)

        # Save plot as PNG
        png_path = os.path.join(out_dir, f"{base_name}_plot.png")
        # Use the same plot as in the GUI
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.scores, color='cyan', linewidth=2, label='Anomaly Score')
        ax.axhline(y=self.threshold_var.get(), color='yellow', linestyle='--', alpha=0.7, label='Threshold')
        ax.set_title('Anomaly Scores')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        # Save summary TXT
        txt_path = os.path.join(out_dir, f"{base_name}_summary.txt")
        threshold = self.threshold_var.get()
        anomaly_frames = [i for i, s in enumerate(self.scores) if s > threshold]
        status = "ANOMALY" if len(anomaly_frames) > 0 else "NORMAL"
        with open(txt_path, 'w') as f:
            f.write(f"Video: {getattr(self, 'video_path', 'N/A')}\n")
            f.write(f"Threshold: {threshold:.6f}\n")
            f.write(f"Total Frames: {len(self.scores)}\n")
            f.write(f"Anomaly Frames: {len(anomaly_frames)}\n")
            f.write(f"Status: {status}\n")
            if anomaly_frames:
                f.write(f"First Anomaly Frame: {anomaly_frames[0]}\n")
                f.write(f"Last Anomaly Frame: {anomaly_frames[-1]}\n")
            f.write(f"\nScores (first 10): {self.scores[:10]}\n")

        self.status_bar.configure(text=f"Results saved to {out_dir}")

    def update_dashboard(self, score):
        self.score_text.configure(text=f"MSE Loss: {score:.5f}")
        
        thresh = self.threshold_var.get()

        if score > thresh:
            self.alert_label.configure(text="🚨 ANOMALY 🚨", text_color="#ff4444")
            self.alert_panel.configure(fg_color="#4a0000") 
        else:
            self.alert_label.configure(text="NORMAL", text_color="#00C851")
            self.alert_panel.configure(fg_color="transparent") 

    def reset_ui(self):
        self.btn_start.configure(state="normal", text="▶ START DETECTION", fg_color="green")
        self.btn_load.configure(state="normal")
        self.btn_save.configure(state="normal")
        self.status_label.configure(text="Analysis Finished", text_color="white")
        self.status_bar.configure(text="Analysis Finished")
        self.alert_label.configure(text="DONE", text_color="gray")
        self.alert_panel.configure(fg_color="transparent")
        self.progress.grid_remove()

if __name__ == "__main__":
    app = AnomalyDetectionApp()
    app.mainloop()