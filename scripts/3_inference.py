#!/usr/bin/env python3
"""
Efficient-VAD — Final inference script (features-only mode, robust)

Key behaviors:
- Works on precomputed feature files: ./data/features_v3/test/**/*.npy (configurable)
- Automatically repairs 1D flattened .npy files when length % INPUT_DIM == 0
- Loads EfficientVAD_Pro checkpoint (handles plain state_dict, wrapped dict, and DataParallel prefixes)
- Batched AE inference, smoothing, center-assignment + linear interpolation mapping
- Dataset-percentile thresholding (configurable) or fixed threshold
- Saves per-video .npz files with keys: per_frame_scores, window_scores, entropy, num_extracted_frames, frame_indices
- Outputs a summary CSV (inference_summary.csv)
- Minimal dependencies: numpy, torch, pandas, tqdm

Usage example:
    python inference_final.py --features ./data/features_v3/test --ckpt ./data/checkpoints/efficient_vad.pth --out ./results/inference

Author: Generated for your Efficient-VAD project
"""

import os
import sys
import glob
import argparse
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EfficientVAD_Pro
from src.config import get_config

# ----------------------------
# Configuration
# ----------------------------
config = get_config()
DEFAULT_FEATURE_DIR = os.path.join(config.paths.features_dir, "test")
DEFAULT_CHECKPOINT = os.path.join(config.paths.checkpoints_dir, "efficient_vad.pth")
DEFAULT_OUTPUT_DIR = os.path.join(config.paths.results_dir, "inference")
DEFAULT_SEQ_LEN = config.inference.sequence_length
DEFAULT_STRIDE = config.inference.stride
DEFAULT_AE_BATCH_GPU = config.inference.batch_size
DEFAULT_AE_BATCH_CPU = 8
DEFAULT_SMOOTH = config.inference.smoothing_kernel
DEFAULT_THRESHOLD = config.inference.fixed_threshold
DEFAULT_PERCENTILE = config.inference.threshold_percentile
INPUT_DIM = config.model.input_dim
HIDDEN_DIM = config.model.hidden_dim
LATENT_DIM = config.model.latent_dim
NUM_MEMORIES = config.model.num_memories
FRAME_SKIP = config.feature_extraction.frame_skip

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_and_fix_feature(path, input_dim=INPUT_DIM):
    """
    Loads a .npy feature file and attempts to fix common shape issues:
    - If 2D with second dim == input_dim -> OK
    - If 1D and length % input_dim == 0 -> reshape to (-1, input_dim)
    - If 2D but shape[0] == input_dim -> transpose attempt
    - If other shapes, try flatten+reshape if divisible
    Returns: feats (np.ndarray shape (N, input_dim)) or None on failure.
    """
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception:
        # try allow_pickle True fallback for legacy files
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[ERROR] Cannot load {path}: {e}")
            return None

    if arr is None:
        return None

    # If already 2D and correct feature dim
    if arr.ndim == 2 and arr.shape[1] == input_dim:
        return arr.astype(np.float32)

    # 1D flattened vector that can be reshaped
    if arr.ndim == 1:
        if arr.size % input_dim == 0 and arr.size > 0:
            N = arr.size // input_dim
            return arr.reshape(N, input_dim).astype(np.float32)
        else:
            return None

    # 2D but transposed or with wrong axis
    if arr.ndim == 2 and arr.shape[0] == input_dim and arr.shape[1] != input_dim:
        # transpose attempt
        new = arr.T
        if new.shape[1] == input_dim:
            return new.astype(np.float32)
        # else cannot fix
        return None

    # 3D or other unexpected shapes: try to flatten and reshape if total size divisible
    total = arr.size
    if total % input_dim == 0 and total > 0:
        N = total // input_dim
        try:
            flat = arr.flatten()
            return flat.reshape(N, input_dim).astype(np.float32)
        except Exception:
            return None

    return None

def sliding_windows(feats, seq_len, stride):
    N = feats.shape[0]
    if N < seq_len:
        return np.empty((0, seq_len, feats.shape[1]), dtype=np.float32), []
    starts = list(range(0, N - seq_len + 1, stride))
    windows = np.stack([feats[s : s + seq_len] for s in starts], axis=0)
    return windows, starts

def remove_dataparallel_prefix(state_dict):
    """
    Removes 'module.' prefix from keys if present (from DataParallel)
    """
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v
    return new_state

def compute_entropy_from_att(att):
    if att is None:
        return None
    a = np.array(att, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=1)
    a = np.clip(a, 1e-12, 1.0)
    ent = -np.sum(a * np.log(a), axis=1)
    return float(ent.mean())

# ----------------------------
# Inference pipeline
# ----------------------------
def run_inference(feature_dir, checkpoint_path, out_dir,
                  seq_len, stride, ae_batch, smooth_kernel,
                  threshold, percentile, device, save_att_full=False):
    ensure_dir(out_dir)
    pattern = os.path.join(feature_dir, "**", "*.npy")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"[INFO] Found {len(files)} feature files under {feature_dir}")

    if len(files) == 0:
        print("[ERROR] No feature files found. Exiting.")
        return

    # Load model
    model = EfficientVAD_Pro(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                             latent_dim=LATENT_DIM, num_memories=NUM_MEMORIES)
    try:
        ck = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return

    if isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck if isinstance(ck, dict) else ck

    # Remove DataParallel prefixes if any
    try:
        state = remove_dataparallel_prefix(state)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[ERROR] Could not load state_dict into model: {e}")
        return

    model.to(device)
    model.eval()
    print("[INFO] Model loaded and ready.")

    all_window_scores = []
    records = []

    # iterate files
    for fpath in tqdm(files, desc="Scoring"):
        fname = os.path.splitext(os.path.basename(fpath))[0]
        feats = load_and_fix_feature(fpath, INPUT_DIM)
        if feats is None or feats.size == 0:
            print(f"[WARN] Skipping {fname}: unable to load or fix shape")
            records.append({
                "feature_path": fpath, "video_name": fname, "num_extracted_frames": 0,
                "max_window_error": 0.0, "avg_window_error": 0.0,
                "entropy": None, "prediction": "INSUFFICIENT"
            })
            continue

        # Build frame indices (assuming extraction started at frame 0 and used FRAME_SKIP)
        num_extracted = feats.shape[0]
        frame_indices = np.arange(0, num_extracted * FRAME_SKIP, FRAME_SKIP, dtype=np.int32)

        windows, starts = sliding_windows(feats, seq_len, stride)
        if windows.shape[0] == 0:
            records.append({
                "feature_path": fpath, "video_name": fname, "num_extracted_frames": int(num_extracted),
                "max_window_error": 0.0, "avg_window_error": 0.0,
                "entropy": None, "prediction": "INSUFFICIENT"
            })
            continue

        # Batched AE inference
        errors = []
        att_list = []
        with torch.no_grad():
            for i in range(0, windows.shape[0], ae_batch):
                batch = torch.from_numpy(windows[i : i + ae_batch]).float().to(device)  # (B, seq, D)
                out = model(batch)
                if isinstance(out, (tuple, list)):
                    recon, att = out[0], out[1]
                else:
                    recon = out
                    att = None
                # per-window mse
                batch_mse = torch.mean((recon - batch) ** 2, dim=[1,2]).cpu().numpy().tolist()
                errors.extend(batch_mse)
                if att is not None:
                    att_list.append(att.detach().cpu().numpy())

        if len(errors) == 0:
            records.append({
                "feature_path": fpath, "video_name": fname, "num_extracted_frames": int(num_extracted),
                "max_window_error": 0.0, "avg_window_error": 0.0,
                "entropy": None, "prediction": "INSUFFICIENT"
            })
            continue

        # smooth and map
        if smooth_kernel and smooth_kernel > 1:
            kernel = int(smooth_kernel)
            window = np.ones(kernel) / kernel
            smoothed = np.convolve(np.array(errors, dtype=np.float32), window, mode="same")
        else:
            smoothed = np.array(errors, dtype=np.float32)

        # map windows -> per-extracted-frame scores
        centers = [s + seq_len // 2 for s in starts]
        x = np.array(centers, dtype=np.float32)
        y = smoothed.astype(np.float32)
        xs = np.arange(num_extracted, dtype=np.float32)
        per_frame_scores = np.interp(xs, x, y, left=y[0], right=y[-1])

        # entropy
        att_array = np.concatenate(att_list, axis=0) if att_list else None
        entropy_val = compute_entropy_from_att(att_array)

        all_window_scores.extend(smoothed.tolist())

        records.append({
            "feature_path": fpath, "video_name": fname, "num_extracted_frames": int(num_extracted),
            "max_window_error": float(np.max(smoothed)), "avg_window_error": float(np.mean(smoothed)),
            "entropy": float(entropy_val) if entropy_val is not None else None, "prediction": None,
            "per_frame_scores": per_frame_scores, "window_scores": smoothed,
            "frame_indices": frame_indices, "att_array": att_array
        })

    # Decide threshold
    if threshold is None:
        if len(all_window_scores) == 0:
            thr = 1e-6
            print("[WARN] No window scores collected; using fallback threshold.")
        else:
            thr = float(np.percentile(np.array(all_window_scores), percentile))
            print(f"[INFO] Using dataset percentile threshold ({percentile}th) = {thr:.8e}")
    else:
        thr = float(threshold)
        print(f"[INFO] Using fixed threshold = {thr:.8e}")

    # Finalize records, save per-video npz and CSV
    out_rows = []
    for rec in records:
        if rec.get("window_scores") is None or len(rec.get("window_scores")) == 0:
            pred = "INSUFFICIENT"
            max_err = 0.0
            avg_err = 0.0
            ent = rec.get("entropy", None)
        else:
            max_err = float(rec["max_window_error"])
            avg_err = float(rec["avg_window_error"])
            ent = rec.get("entropy", None)
            pred = "ABNORMAL" if max_err > thr else "NORMAL"

            # Save per-video .npz
            base = os.path.splitext(os.path.basename(rec["feature_path"]))[0]
            save_path = os.path.join(out_dir, base + ".npz")
            try:
                np.savez_compressed(save_path,
                                    per_frame_scores=rec["per_frame_scores"].astype(np.float32),
                                    window_scores=rec["window_scores"].astype(np.float32),
                                    entropy=np.array([ent], dtype=np.float32) if ent is not None else np.array([], dtype=np.float32),
                                    num_extracted_frames=np.array([rec["num_extracted_frames"]], dtype=np.int32),
                                    frame_indices=rec["frame_indices"].astype(np.int32))
                # optionally save full att array (disabled by default)
                if save_att_full and rec.get("att_array") is not None:
                    att_path = os.path.join(out_dir, base + "_att.npy")
                    np.save(att_path, rec["att_array"].astype(np.float32))
            except Exception as e:
                print(f"[WARN] Failed to save per-video scores for {base}: {e}")

        out_rows.append({
            "feature_path": rec["feature_path"],
            "video_name": rec["video_name"],
            "num_extracted_frames": int(rec["num_extracted_frames"]),
            "max_window_error": max_err,
            "avg_window_error": avg_err,
            "entropy": ent if ent is not None else None,
            "prediction": pred
        })

    # write CSV
    df = pd.DataFrame(out_rows)
    csv_path = os.path.join(out_dir, "inference_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Inference finished. Summary CSV: {csv_path}")
    print(f"[INFO] Per-video .npz files saved under: {out_dir}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient-VAD inference (features-only)")
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURE_DIR, help="Feature dir (recursive .npy)")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--ae_batch", type=int, default=None, help="AE batch size (auto by device if not set)")
    parser.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fixed threshold; if omitted percentile used")
    parser.add_argument("--percentile", type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--frame_skip", type=int, default=FRAME_SKIP, help="Frame skip used during feature extraction")
    parser.add_argument("--save_att", action="store_true", help="Save full attention arrays (may be large)")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_batch = args.ae_batch if args.ae_batch is not None else (DEFAULT_AE_BATCH_GPU if DEVICE.type == "cuda" else DEFAULT_AE_BATCH_CPU)

    # update global constants used in mapping
    FRAME_SKIP = int(args.frame_skip)

    run_inference(feature_dir=args.features,
                  checkpoint_path=args.ckpt,
                  out_dir=args.out,
                  seq_len=args.seq_len,
                  stride=args.stride,
                  ae_batch=ae_batch,
                  smooth_kernel=args.smooth,
                  threshold=args.threshold,
                  percentile=args.percentile,
                  device=DEVICE,
                  save_att_full=args.save_att)