"""
main.py - Entry point for Efficient-VAD Thesis Project

This script runs the complete anomaly detection pipeline step by step:
1. Feature Extraction
2. Model Training
3. Inference (Anomaly Scoring)
4. Evaluation (Metrics & Plots)

You can skip any step using command-line arguments.

Usage:
    python main.py [--skip-extraction] [--skip-training] [--skip-inference] [--skip-evaluation]

Example:
    python main.py --skip-extraction --skip-training

Author: Your Name
Date: 2025-12-14
"""
import sys
import os
import argparse
import subprocess

PIPELINE_SCRIPT = os.path.join(os.path.dirname(__file__), 'scripts', 'run_pipeline.py')

def main():
    parser = argparse.ArgumentParser(description="Efficient-VAD: Full Pipeline Runner")
    parser.add_argument('--skip-extraction', action='store_true', help='Skip feature extraction')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-inference', action='store_true', help='Skip inference')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation')
    args = parser.parse_args()

    cmd = [sys.executable, PIPELINE_SCRIPT]
    if args.skip_extraction:
        cmd.append('--skip-extraction')
    if args.skip_training:
        cmd.append('--skip-training')
    if args.skip_inference:
        cmd.append('--skip-inference')
    if args.skip_evaluation:
        cmd.append('--skip-evaluation')

    print("\n[INFO] Running Efficient-VAD Full Pipeline...\n")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("\n[SUCCESS] Pipeline completed successfully.")
    else:
        print(f"\n[ERROR] Pipeline failed with exit code {result.returncode}.")

if __name__ == "__main__":
    main()
