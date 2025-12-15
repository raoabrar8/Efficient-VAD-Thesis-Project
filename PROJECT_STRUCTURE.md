# Project Structure

This document describes the organization of the Efficient-VAD thesis project.

## Directory Structure

```
Efficient-VAD Thesis Project/
│
├── config.yaml                    # Main configuration file (YAML)
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup script
├── README.md                      # Main project documentation
├── PROJECT_STRUCTURE.md           # This file
├── .gitignore                     # Git ignore rules
│
├── scripts/                       # Main execution scripts
│   ├── 1_extract_features.py     # Step 1: Feature extraction
│   ├── 2_train.py                # Step 2: Model training
│   ├── 3_inference.py            # Step 3: Anomaly detection
│   ├── 4_evaluate.py             # Step 4: Evaluation and metrics
│   ├── run_pipeline.py           # Run complete pipeline
│   ├── 6_benchmark_speed.py      # Speed benchmarking (optional)
│   ├── 7_visualize_tsne.py      # t-SNE visualization (optional)
│   └── utils/                    # Utility scripts
│       ├── check_features.py     # Feature file validation
│       ├── inspect_npy_details.py # Feature inspection
│       ├── reextract_missing_features.py # Re-extract missing features
│       └── test_extract_features.py # Test feature extraction
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── model.py                  # Model architecture (EfficientVAD_Pro)
│   ├── dataset.py                # Dataset loaders
│   └── utils.py                  # Utility functions
│
├── data/                         # Data directory
│   ├── normal_videos/           # Training videos (normal only)
│   ├── test_videos/             # Test videos (normal + abnormal)
│   │   └── UCF-Crime/          # UCF-Crime dataset structure
│   │       ├── Normal/
│   │       ├── Abuse/
│   │       ├── Arrest/
│   │       └── ...
│   ├── features_v3/            # Extracted features
│   │   ├── train/             # Training features
│   │   └── test/              # Test features
│   │       └── UCF-Crime/
│   │           ├── Normal/
│   │           ├── Abuse/
│   │           └── ...
│   └── checkpoints/           # Saved models
│       └── efficient_vad.pth  # Trained model
│
├── results/                     # Results and outputs
│   ├── inference/              # Inference results
│   │   ├── *.npz              # Per-video scores
│   │   └── inference_summary.csv
│   └── evaluation/            # Evaluation results
│       ├── *.png              # Plots and graphs
│       └── results.txt        # Metrics report
│
└── logs/                       # Training logs
    └── training_log.csv       # Training history
```

## File Descriptions

### Configuration
- **config.yaml**: Centralized configuration for all components
- **src/config.py**: Configuration loader and manager

### Main Scripts
- **scripts/1_extract_features.py**: Extracts MobileNetV3 features from videos
- **scripts/2_train.py**: Trains the LSTM Autoencoder with Memory Module
- **scripts/3_inference.py**: Runs anomaly detection on test videos
- **scripts/4_evaluate.py**: Generates evaluation metrics and visualizations
- **scripts/run_pipeline.py**: Runs the complete pipeline end-to-end

### Source Code
- **src/model.py**: EfficientVAD_Pro model architecture
  - MemoryModule: Stores normal pattern prototypes
  - EfficientVAD_Pro: LSTM Autoencoder with memory augmentation
- **src/dataset.py**: Dataset loaders for training and inference
- **src/utils.py**: Utility functions (smoothing, normalization, etc.)

### Data
- **data/normal_videos/**: Training videos (normal behavior only)
- **data/test_videos/**: Test videos (normal + 13 anomaly classes)
- **data/features_v3/**: Extracted features (`.npy` files)
- **data/checkpoints/**: Saved model checkpoints

### Results
- **results/inference/**: Per-video anomaly scores and summaries
- **results/evaluation/**: Evaluation metrics, plots, and reports

## Workflow

1. **Feature Extraction**: `scripts/1_extract_features.py`
   - Processes videos → extracts MobileNetV3 features
   - Output: `data/features_v3/train/` and `data/features_v3/test/`

2. **Training**: `scripts/2_train.py`
   - Trains on normal features only (self-supervised)
   - Output: `data/checkpoints/efficient_vad.pth`

3. **Inference**: `scripts/3_inference.py`
   - Runs anomaly detection on test features
   - Output: `results/inference/*.npz` and `inference_summary.csv`

4. **Evaluation**: `scripts/4_evaluate.py`
   - Calculates metrics (AUC, Accuracy, etc.)
   - Generates visualizations
   - Output: `results/evaluation/`

## Quick Start

Run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

Or run steps individually:
```bash
python scripts/1_extract_features.py
python scripts/2_train.py
python scripts/3_inference.py
python scripts/4_evaluate.py
```

## Notes

- All paths are configurable via `config.yaml`
- The project uses relative paths from the project root
- Features are stored as NumPy arrays (`.npy` files)
- Models are saved as PyTorch state dictionaries (`.pth` files)



