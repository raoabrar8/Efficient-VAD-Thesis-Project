# Chapter 3: Methodology and Experimental Setup

## 3.1 Project Overview
Efficient-VAD is a self-supervised anomaly detection system for surveillance videos, leveraging a MobileNetV3-based feature extractor and an LSTM autoencoder with a memory module. The system is designed for real-time, efficient, and accurate anomaly detection, with a user-friendly GUI for visualization and analysis.

## 3.1.1 Implemented Features and Improvements

This project includes a comprehensive set of features, automation, and improvements, implemented across the pipeline, codebase, and GUI:

- **Self-Supervised Learning:**
  - Training uses only normal videos, requiring no labeled anomaly data.
  - Memory-augmented LSTM autoencoder learns normal patterns and reconstructs them, with high error on anomalies.

- **Efficient Feature Extraction:**
  - MobileNetV3-Large backbone for fast, low-resource feature extraction.
  - Frame skipping and batch processing for speed.
  - Automatic repair of flattened/incorrect .npy feature files.

- **Robust Model Architecture:**
  - LSTM encoder-decoder with memory module (50 prototypes).
  - Handles DataParallel checkpoints and state_dict variations.
  - Configurable via YAML (`config.yaml`).

- **Pipeline Automation:**
  - One-click pipeline via `main.py` or `scripts/run_pipeline.py`.
  - Modular scripts for each step: feature extraction, training, inference, evaluation, benchmarking, t-SNE, and visualization.
  - Command-line flags to skip or repeat steps.

- **Advanced Evaluation and Metrics:**
  - Computes AUC, accuracy, F1, precision, recall, confusion matrix, and classification report.
  - Frame-level, window-level, and video-level evaluation.
  - ROC and PR curve generation.
  - Frame-level anomaly plots with ground truth overlays.
  - t-SNE visualizations of features and scores.
  - Efficiency analysis: FPS, memory usage, model size, parameter count.
  - Benchmarking script for speed and resource profiling.

- **Threshold Optimization:**
  - Supports fixed, percentile-based, and data-driven thresholds (mean+std, grid search, histogram analysis).
  - GUI and script-based threshold tuning.
  - Auto-tune and best-threshold selection in GUI.

- **Ground Truth Generation:**
  - Automated frame-level ground truth CSV generation for evaluation and visualization.
  - Exports per-video and per-frame results for reproducibility.

- **Comprehensive Logging and Export:**
  - All results, metrics, and plots are saved to organized directories (`results/`, `results_graphs_v3/`).
  - Exports: CSV (scores, labels, summary), PNG (plots), TXT (logs, reports).
  - Inference summary and per-video .npz files for downstream analysis.

- **GUI (CustomTkinter):**
  - Real-time anomaly score plotting and video playback.
  - Adjustable threshold and anomaly ratio sliders.
  - Auto-tune and best-threshold buttons.
  - Export of results (CSV, PNG, TXT) directly from GUI.
  - Frame-level anomaly visualization and overlay.
  - Robust error handling and user feedback.

- **Visualization and Reporting:**
  - Score distributions, ROC/PR curves, confusion matrix, t-SNE plots.
  - Frame-level anomaly plots with shaded regions for detected anomalies.
  - Efficiency and benchmarking plots for thesis-ready figures.
  - All plots saved at publication quality (high DPI, tight layout).

- **Reproducibility and Code Quality:**
  - All scripts and configs are version-controlled.
  - Modular code structure (`src/`, `scripts/`, `utils/`).
  - Automated feature quality checker for data integrity.
  - Clear directory structure and documentation (`README.md`, `PROJECT_STRUCTURE.md`).

- **Extensibility and Future-Proofing:**
  - Easy to add new datasets, models, or evaluation metrics.
  - Configurable paths, hyperparameters, and evaluation settings.
  - Modular design for research and deployment.

All of these features and improvements were implemented and validated as part of this thesis project, ensuring a robust, efficient, and reproducible anomaly detection pipeline for surveillance video analysis.

## 3.2 Dataset
- **UCF-Crime** dataset structure:
  - **Training:** Normal videos only (self-supervised)
  - **Testing:** Normal + 13 anomaly classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism)
- Directory structure and data splits are detailed in the project documentation.

## 3.3 Project Structure
- `config.yaml`: Central configuration
- `requirements.txt`: Dependencies
- `scripts/`: Main pipeline scripts (feature extraction, training, inference, evaluation, benchmarking, visualization)
- `src/`: Source code (model, dataset, utils)
- `data/`: Video and feature storage
- `results/`: Inference, evaluation, and visualization outputs
- `gui.py`: CustomTkinter-based GUI for interactive analysis

## 3.4 Model Architecture
- **Feature Extraction:**
  - MobileNetV3-Large backbone, pretrained on ImageNet
  - Input: 224x224 video frames
  - Output: 960-dim feature vectors
- **Anomaly Detection Model:**
  - LSTM-based encoder (960→256→64)
  - Memory module (50 normal pattern prototypes)
  - LSTM-based decoder (64→256→960)
  - Loss: MSE reconstruction + entropy regularization
- **Self-Supervised Learning:**
  - Trained only on normal videos
  - Learns to reconstruct normal patterns; fails on anomalies (high error)

## 3.5 Training Procedure
- **Feature Extraction:**
  - `scripts/1_extract_features.py` processes all videos, saving features to `data/features_v3/`
- **Model Training:**
  - `scripts/2_train.py` trains the LSTM autoencoder on normal features
  - Hyperparameters: batch size, epochs, learning rate, sequence length, memory size, entropy/sparsity weights (see `config.yaml`)
  - Loss: MSE + entropy + sparsity
- **Checkpointing:**
  - Best model saved to `data/checkpoints/efficient_vad.pth`

## 3.6 Inference and Evaluation
- **Inference:**
  - `scripts/3_inference.py` runs anomaly detection on test features
  - Outputs per-frame anomaly scores to `results/FrameLevel_Results/`
- **Evaluation:**
  - `scripts/4_evaluate.py` computes metrics (AUC, accuracy, F1, precision, recall)
  - Generates ROC, PR, and frame-level anomaly plots
  - Efficiency analysis: FPS, memory, model size, params
  - t-SNE visualization: `scripts/7_visualize_tsne.py`
- **Threshold Selection:**
  - Default threshold: 0.22 (mean + 1 std from anomaly score distribution)
  - Threshold can be tuned via GUI or config

## 3.7 GUI and Visualization
- **GUI (`gui.py`):**
  - CustomTkinter-based, real-time anomaly score plotting
  - Adjustable threshold and anomaly ratio
  - Auto-tune and best-threshold features
  - Results export (CSV, PNG, TXT)
- **Visualization:**
  - Score distributions, ROC/PR curves, t-SNE plots
  - Efficiency and benchmarking plots

## 3.8 Experimental Setup
- **Hardware:**
  - GPU: NVIDIA (if available), otherwise CPU
  - RAM: 8GB+
- **Software:**
  - Python 3.8+
  - PyTorch, torchvision, numpy, pandas, matplotlib, scikit-learn, customtkinter
  - All dependencies in `requirements.txt`
- **Reproducibility:**
  - All scripts and configs version-controlled
  - Results and logs saved in `results/`

## 3.9 Pipeline Automation
- **Full pipeline:**
  - `main.py` or `scripts/run_pipeline.py` for one-click execution
  - Steps: feature extraction → training → inference → evaluation
  - Command-line flags to skip steps as needed

## 3.10 Limitations and Future Work
- **Limitations:**
  - Threshold selection is data-dependent
  - Model may miss subtle anomalies or produce false positives on rare normal patterns
  - No explicit temporal localization of anomalies
- **Future Work:**
  - Explore adaptive thresholding, multi-modal features, and more robust memory modules
  - Integrate video annotation and feedback in GUI
  - Extend to other datasets and real-world deployments

---

*This chapter provides a comprehensive methodology and experimental setup for Efficient-VAD, ensuring reproducibility and clarity for thesis documentation.*
