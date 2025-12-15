#!/usr/bin/env python3
"""
Main pipeline script to run the complete Efficient-VAD workflow.
This script runs all steps sequentially: feature extraction, training, inference, and evaluation.
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False
    print(f"\n[SUCCESS] {description} completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Efficient-VAD pipeline")
    parser.add_argument("--skip-extraction", action="store_true", 
                       help="Skip feature extraction (use existing features)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (use existing model)")
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference (use existing results)")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation")
    
    args = parser.parse_args()
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(scripts_dir)
    
    steps = []
    
    # Step 1: Feature Extraction
    if not args.skip_extraction:
        steps.append((
            [sys.executable, os.path.join(scripts_dir, "1_extract_features.py")],
            "Feature Extraction"
        ))
    
    # Step 2: Training
    if not args.skip_training:
        steps.append((
            [sys.executable, os.path.join(scripts_dir, "2_train.py")],
            "Model Training"
        ))
    
    # Step 3: Inference
    if not args.skip_inference:
        steps.append((
            [sys.executable, os.path.join(scripts_dir, "3_inference.py")],
            "Anomaly Detection Inference"
        ))
    
    # Step 4: Evaluation
    if not args.skip_evaluation:
        steps.append((
            [sys.executable, os.path.join(scripts_dir, "4_evaluate.py")],
            "Evaluation and Metrics"
        ))
    
    if not steps:
        print("[WARNING] All steps are skipped. Nothing to do.")
        return
    
    print("\n" + "="*60)
    print("EFFICIENT-VAD PIPELINE")
    print("="*60)
    print(f"Total steps: {len(steps)}")
    print("="*60)
    
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {desc}")
        if not run_command(cmd, desc):
            print(f"\n[ERROR] Pipeline stopped at step {i}: {desc}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults are available in:")
    print(f"  - Model: {os.path.join(base_dir, 'data/checkpoints/efficient_vad.pth')}")
    print(f"  - Inference: {os.path.join(base_dir, 'results/inference/')}")
    print(f"  - Evaluation: {os.path.join(base_dir, 'results/evaluation/')}")
    print("="*60)

if __name__ == "__main__":
    main()



