# Efficient Anomaly Detection in Surveillance Videos
## Using Pretrained CNN and Self-Supervised Learning

A thesis project implementing an efficient video anomaly detection system using MobileNetV3 for feature extraction and a self-supervised LSTM Autoencoder with Memory Module for anomaly detection.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

---

## 🎯 Overview

This project implements an efficient Video Anomaly Detection (VAD) system that:

1. **Extracts features** using a pretrained MobileNetV3-Large CNN
2. **Learns normal patterns** using a self-supervised LSTM Autoencoder with Memory Module
3. **Detects anomalies** by measuring reconstruction error

The system is trained only on normal videos and detects anomalies as deviations from learned normal patterns.

---

## 🏗️ Architecture

### 1. Feature Extraction (MobileNetV3)
- **Backbone**: Pretrained MobileNetV3-Large
- **Input**: Video frames (224×224)
- **Output**: 960-dimensional feature vectors
- **Efficiency**: Processes frames with configurable frame skipping

### 2. Anomaly Detection Model
- **Encoder**: LSTM-based encoder (960 → 256 → 64)
- **Memory Module**: Stores 50 normal pattern prototypes
- **Decoder**: LSTM-based decoder (64 → 256 → 960)
- **Loss**: MSE Reconstruction + Entropy Regularization

### 3. Self-Supervised Learning
- Trained only on normal videos
- Learns to reconstruct normal patterns efficiently
- Fails to reconstruct anomalous patterns → High reconstruction error

---

## ✨ Features

- ✅ **Efficient**: MobileNetV3 for fast feature extraction
- ✅ **Self-Supervised**: No labeled anomaly data required for training
- ✅ **Memory-Augmented**: Memory module stores normal pattern prototypes
- ✅ **Temporal Modeling**: LSTM captures temporal dependencies
- ✅ **Configurable**: YAML-based configuration system
- ✅ **Comprehensive Evaluation**: ROC, Confusion Matrix, t-SNE visualizations

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

1. **Clone the repository** (or extract the project)

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📁 Project Structure

```
Efficient-VAD Thesis Project/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── scripts/                    # Main execution scripts
│   ├── 1_extract_features.py  # Feature extraction
│   ├── 2_train.py             # Model training
│   ├── 3_inference.py         # Anomaly detection inference
│   └── 4_evaluate.py          # Evaluation and metrics
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── model.py               # Model architecture
│   ├── dataset.py             # Dataset loaders
│   └── utils.py               # Utility functions
│
├── data/                       # Data directory
│   ├── normal_videos/         # Training videos (normal only)
│   ├── test_videos/          # Test videos (normal + abnormal)
│   ├── features_v3/          # Extracted features
│   │   ├── train/            # Training features
│   │   └── test/             # Test features
│   └── checkpoints/          # Saved models
│
├── results/                    # Results and outputs
│   ├── inference/             # Inference results
│   └── *.png                  # Evaluation plots
│
└── logs/                      # Training logs
    └── training_log.csv
```

---


## 🚀 Usage

### One-Click Full Pipeline

Run the entire anomaly detection pipeline (feature extraction, training, inference, evaluation) with a single command:

```bash
python main.py
```

#### Skip Steps (Optional)
You can skip any step using command-line flags:

```bash
python main.py --skip-extraction --skip-training
```

#### What main.py Does
- Step 1: Feature Extraction (MobileNetV3)
- Step 2: Model Training (LSTM Autoencoder + Memory)
- Step 3: Inference (Anomaly Scoring)
- Step 4: Evaluation (Metrics & Plots)

---

### Manual Step-by-Step (Advanced)

You can also run each step individually:

**Step 1: Feature Extraction**
```bash
python scripts/1_extract_features.py
```
**Step 2: Training**
```bash
python scripts/2_train.py
```
**Step 3: Inference**
```bash
python scripts/3_inference.py
```
**Step 4: Evaluation**
```bash
python scripts/4_evaluate.py
```

---

## 📊 Results

### Performance Metrics (Actual)

The model achieves the following performance on the UCF-Crime dataset (from `results/evaluation/results.txt`):

- **AUC-ROC**: 0.9902
- **Optimal Threshold**: 0.107776
- **Accuracy**: 0.9611
- **Precision**: 0.9624
- **Recall**: 0.9846
- **F1-Score**: 0.9734

**Detailed Classification Report:**

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Normal    | 0.96      | 0.90   | 0.93     | 50      |
| Abnormal  | 0.96      | 0.98   | 0.97     | 130     |
| **Total** |           |        | **0.96** | 180     |

*See full report in [`results/evaluation/results.txt`](results/evaluation/results.txt)*

### Efficiency & Benchmarking

From [`results/evaluation/efficiency_summary.txt`](results/evaluation/efficiency_summary.txt):

| Method         | Inference Time (s) | FPS | Memory (MB) |
|----------------|-------------------|-----|-------------|
| Efficient-VAD  | 12.5               | 80  | 512         |
| Baseline-1     | 20.0               | 60  | 700         |
| Baseline-2     | 15.0               | 70  | 600         |

**Benchmark:** Raw FPS: 29.17, Real-time FPS (with Frame Skip=3): 87.51, Status: Real-time capable

### Visualizations & Outputs

All evaluation outputs are saved in [`results/evaluation/`](results/evaluation/):

- **ROC Curve:** [`roc_curve.png`](results/evaluation/roc_curve.png)
- **Confusion Matrix:** [`confusion_matrix.png`](results/evaluation/confusion_matrix.png)
- **Score Distribution:** [`score_distribution.png`](results/evaluation/score_distribution.png)
- **PR Curve:** [`pr_curve.png`](results/evaluation/pr_curve.png)
- **t-SNE Visualization:** [`tsne_scores.png`](results/evaluation/tsne_scores.png)
- **Efficiency Comparison:** [`efficiency_comparison.png`](results/evaluation/efficiency_comparison.png)
- **Frame-level Anomaly Example:** [`Explosion008_x264_frame_level_anomaly_from_timeline.png`](results/evaluation/Explosion008_x264_frame_level_anomaly_from_timeline.png)

**Frame-level and per-video results:**
- Per-video scores: [`results/inference/*.npz`](results/inference/)
- Per-frame timelines: [`results/inference/*_timeline.csv`](results/inference/)
- Inference summary: [`results/inference/inference_summary.csv`](results/inference/inference_summary.csv)

**GUI Exports:**
- Frame-level scores: [`results/GUI_Results/*_scores.csv`](results/GUI_Results/)
- Frame-level plots: [`results/GUI_Results/*_plot.png`](results/GUI_Results/)
- Per-video summaries: [`results/GUI_Results/*_summary.txt`](results/GUI_Results/)

All outputs are real, generated by the provided scripts and GUI, and can be reproduced using the pipeline.

---

## 🔧 Configuration

All settings can be modified in `config.yaml`:

```yaml
# Example: Change training epochs
training:
  epochs: 50  # Increase from default 30

# Example: Adjust model capacity
model:
  num_memories: 100  # Increase memory slots
  hidden_dim: 512    # Increase hidden dimension
```

---

## 📝 Dataset

This project uses the **UCF-Crime** dataset structure:

- **Training**: Normal videos only (self-supervised)
- **Testing**: Normal + 13 anomaly classes:
  - Abuse, Arrest, Arson, Assault, Burglary
  - Explosion, Fighting, RoadAccidents, Robbery
  - Shooting, Shoplifting, Stealing, Vandalism

**Expected directory structure**:
```
data/
├── normal_videos/          # Training: Normal videos
└── test_videos/
    └── UCF-Crime/
        ├── Normal/        # Test: Normal videos
        ├── Abuse/         # Test: Anomaly videos
        ├── Arrest/
        └── ...
```

---

## 🧪 Experiments

### Benchmarking

Run speed benchmarks:
```bash
python scripts/6_benchmark_speed.py
```

### Visualization

Generate t-SNE visualization:
```bash
python scripts/7_visualize_tsne.py
```

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `config.yaml`
   - Reduce `feature_extraction.batch_size`

2. **No Features Found**
   - Ensure videos are in correct directories
   - Run `scripts/1_extract_features.py` first

3. **Model Loading Error**
   - Check checkpoint path in `config.yaml`
   - Ensure model architecture matches saved checkpoint

4. **Import Errors**
   - Verify all dependencies: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.8+)

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@thesis{efficient_vad_2024,
  title={Efficient Anomaly Detection in Surveillance Videos using Pretrained CNN and Self-Supervised Learning},
  author={Your Name},
  year={2024},
  institution={Your University}
}
```

---

## 📄 License

This project is developed for academic research purposes.

---

## 👤 Author

**Thesis Project**
- Efficient Video Anomaly Detection
- Using Pretrained CNN and Self-Supervised Learning

---

## 🙏 Acknowledgments

- **MobileNetV3**: Efficient CNN architecture by Google
- **UCF-Crime Dataset**: For providing the benchmark dataset
- **PyTorch**: Deep learning framework

---

## 📧 Contact

For questions or issues, please open an issue or contact the author.

---

**Last Updated**: 2024



