# Quick Start Guide

Get started with the Efficient-VAD project in minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Quick Run

### Option 1: Run Complete Pipeline
```bash
python scripts/run_pipeline.py
```

This will run all steps:
1. Feature extraction (if needed)
2. Model training
3. Inference
4. Evaluation

### Option 2: Run Steps Individually

**Step 1: Extract Features** (if not already done)
```bash
python scripts/1_extract_features.py
```

**Step 2: Train Model**
```bash
python scripts/2_train.py
```

**Step 3: Run Inference**
```bash
python scripts/3_inference.py
```

**Step 4: Evaluate**
```bash
python scripts/4_evaluate.py
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture (hidden dimensions, memory slots)
- Training parameters (epochs, batch size, learning rate)
- Paths to data directories
- Inference settings

## Expected Outputs

After running the pipeline:

- **Model**: `data/checkpoints/efficient_vad.pth`
- **Inference Results**: `results/inference/inference_summary.csv`
- **Evaluation Plots**: `results/evaluation/*.png`
- **Metrics Report**: `results/evaluation/results.txt`

## Troubleshooting

### "No features found"
- Run `scripts/1_extract_features.py` first
- Check that videos are in `data/normal_videos/` and `data/test_videos/`

### "CUDA out of memory"
- Reduce `batch_size` in `config.yaml`
- Reduce `feature_extraction.batch_size` for feature extraction

### "Model not found"
- Ensure training completed successfully
- Check `data/checkpoints/efficient_vad.pth` exists

## Next Steps

- Read `README.md` for detailed documentation
- Check `PROJECT_STRUCTURE.md` for project organization
- Review `config.yaml` for all configuration options



