def efficiency_analysis():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    output_dir = OUTPUT_GRAPH_DIR
    # Read efficiency.csv
    eff_csv = os.path.join(config.paths.results_dir, 'efficiency.csv')
    bench_csv = os.path.join(config.paths.results_dir, 'benchmark', 'benchmark_results.csv')
    summary_lines = []
    if os.path.exists(eff_csv):
        eff_df = pd.read_csv(eff_csv)
        print("\n=== Efficiency Analysis (Per-Method) ===")
        print(eff_df.to_string(index=False))
        summary_lines.append("Efficiency Analysis (Per-Method):\n" + eff_df.to_string(index=False) + "\n")
        # Bar plot: FPS and Memory
        fig, ax1 = plt.subplots(figsize=(8,5), dpi=200)
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        eff_df.plot(x='method', y='fps', kind='bar', ax=ax1, color=color1, legend=False, position=0, width=0.4)
        ax1.set_ylabel('FPS', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax2 = ax1.twinx()
        eff_df.plot(x='method', y='memory', kind='bar', ax=ax2, color=color2, legend=False, position=1, width=0.4)
        ax2.set_ylabel('Memory (MB)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_xticklabels(eff_df['method'], rotation=15)
        plt.title('Efficiency Comparison: FPS and Memory')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved efficiency comparison plot to {output_dir}/efficiency_comparison.png")
    else:
        print("[WARN] efficiency.csv not found.")
    # Read benchmark_results.csv
    if os.path.exists(bench_csv):
        bench_df = pd.read_csv(bench_csv)
        print("\n=== Benchmark Results ===")
        print(bench_df.to_string(index=False))
        summary_lines.append("Benchmark Results:\n" + bench_df.to_string(index=False) + "\n")
    else:
        print("[WARN] benchmark_results.csv not found.")
    # Save summary to file
    if summary_lines:
        with open(os.path.join(output_dir, 'efficiency_summary.txt'), 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"[INFO] Saved efficiency summary to {output_dir}/efficiency_summary.txt")
def plot_frame_level_from_timeline(timeline_csv, label_csv, output_dir):
    import pandas as pd
    s_df = pd.read_csv(timeline_csv)
    plt.figure(figsize=(12,4), dpi=300)
    plt.plot(s_df['frame'], s_df['score'], label='Anomaly Score', color='c')
    # Shade regions where score > 0.18
    frames = s_df['frame'].values
    scores = s_df['score'].values
    threshold = 0.18
    in_region = False
    region_start = None
    anomaly_mask = (scores > threshold).astype(int)
    # Overlay anomaly mask as a step plot (binary, below the main curve)
    plt.step(frames, anomaly_mask, where='post', color='r', alpha=0.7, label='Anomaly Frames')
    # Also shade regions for visual clarity
    for i, val in enumerate(scores):
        if val > threshold and not in_region:
            in_region = True
            region_start = frames[i]
        elif val <= threshold and in_region:
            in_region = False
            region_end = frames[i-1]
            plt.axvspan(region_start, region_end, color='r', alpha=0.2)
    if in_region:
        plt.axvspan(region_start, frames[-1], color='r', alpha=0.2)
    plt.xlabel('Frame Index')
    plt.ylabel('Anomaly Score')
    plt.title(f'Frame-level Anomaly Score: {os.path.basename(timeline_csv).replace('_timeline.csv','')}')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{os.path.basename(timeline_csv).replace('_timeline.csv','')}_frame_level_anomaly_from_timeline.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved frame-level anomaly score plot to {out_path}")
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.interpolate import interp1d

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_config

# ==========================================
# CONFIGURATION
# ==========================================
config = get_config()
SCORE_DIR = os.path.join(config.paths.features_dir, "test")
OUTPUT_GRAPH_DIR = os.path.join(config.paths.results_dir, "evaluation")
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# Ensure results directory exists
os.makedirs(config.paths.results_dir, exist_ok=True)

# Define what counts as "Abnormal" based on folder names
ABNORMAL_CLASSES = config.evaluation.abnormal_classes

def evaluate():
    # 10. Frame-level Anomaly Score vs Frame Index (for at least one abnormal video)
    # Try to find a per-frame scores/labels CSV for an abnormal video
    import glob
    gui_results_dir = os.path.join(config.paths.results_dir, 'GUI_Results')
    frame_score_files = glob.glob(os.path.join(gui_results_dir, '*_scores.csv'))
    frame_label_files = glob.glob(os.path.join(gui_results_dir, '*_labels.csv'))
    found = False
    for score_file in frame_score_files:
        base = os.path.basename(score_file).replace('_scores.csv','')
        label_file = os.path.join(gui_results_dir, f'{base}_labels.csv')
        print(f"[DEBUG] Checking: {score_file} and {label_file}")
        if os.path.exists(label_file):
            # Check if abnormal (majority label==1)
            s_df = None
            l_df = None
            try:
                import pandas as pd
                s_df = pd.read_csv(score_file)
                l_df = pd.read_csv(label_file)
                print(f"[DEBUG] {base}: len(s_df)={len(s_df)}, len(l_df)={len(l_df)}, sum(labels)={l_df['label'].sum()}")
                if len(s_df) == len(l_df) and l_df['label'].sum() > 0:
                    found = True
                    plt.figure(figsize=(12,4), dpi=200)
                    plt.plot(s_df['frame'], s_df['score'], label='Anomaly Score', color='c')
                    # Only fill red where label==1 (contiguous regions)
                    labels = l_df['label'].values
                    frames = s_df['frame'].values
                    in_region = False
                    region_start = None
                    for i, val in enumerate(labels):
                        if val == 1 and not in_region:
                            in_region = True
                            region_start = frames[i]
                        elif val == 0 and in_region:
                            in_region = False
                            region_end = frames[i-1]
                            plt.axvspan(region_start, region_end, color='r', alpha=0.2)
                    if in_region:
                        plt.axvspan(region_start, frames[-1], color='r', alpha=0.2)
                    plt.xlabel('Frame Index')
                    plt.ylabel('Anomaly Score')
                    plt.title(f'Frame-level Anomaly Score: {base}')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, f'{base}_frame_level_anomaly.png'), bbox_inches='tight')
                    plt.close()
                    print(f"[INFO] Saved frame-level anomaly score plot for {base}.")
                    break
            except Exception as e:
                print(f"[WARN] Could not plot frame-level anomaly for {base}: {e}")
    if not found:
        print("[WARN] No abnormal video with per-frame scores/labels found for frame-level plot.")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.manifold import TSNE
    print("[INFO] Loading inference results...")

    y_true = []   # 0 = Normal, 1 = Abnormal
    y_scores = [] # Max anomaly score for each video

    # Try to load from inference results first (preferred)
    inference_dir = os.path.join(config.paths.results_dir, "inference")
    inference_summary = os.path.join(inference_dir, "inference_summary.csv")

    if os.path.exists(inference_summary):
        # Load from inference summary CSV
        import pandas as pd
        print(f"[INFO] Loading from inference summary: {inference_summary}")
        df = pd.read_csv(inference_summary)

        for _, row in df.iterrows():
            video_name = row['video_name']
            max_score = row['max_window_error']

            # Determine label from feature path or video name
            feature_path = str(row.get('feature_path', ''))
            norm_path = feature_path.replace("\\", "/")
            is_abnormal = any(cls in norm_path or cls in video_name for cls in ABNORMAL_CLASSES)

            y_true.append(1 if is_abnormal else 0)
            y_scores.append(max_score)
    else:
        # Fallback: Try to load from feature directory (old method)
        print("[INFO] Inference summary not found. Trying to load from feature files...")
        score_files = glob.glob(os.path.join(SCORE_DIR, "**", "*.npy"), recursive=True)

        if len(score_files) == 0:
            print("[ERROR] No scores found! Please run 'scripts/3_inference.py' first.")
            return

        print(f"[INFO] Found {len(score_files)} score files. Calculating metrics...")

        # Process each file
        for f_path in score_files:
            try:
                scores = np.load(f_path)
                if len(scores) == 0: continue

                # Metric: We take the MAX score of the video.
                video_score = np.max(scores)

                # Determine Label
                norm_path = f_path.replace("\\", "/")
                is_abnormal = any(cls in norm_path for cls in ABNORMAL_CLASSES)

                y_true.append(1 if is_abnormal else 0)
                y_scores.append(video_score)

            except Exception as e:
                print(f"Error reading {f_path}: {e}")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if len(y_true) == 0:
        print("[ERROR] No valid data found.")
        return

    # 3. Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 4. Find Optimal Threshold (Youden's J statistic)
    # Maximize (True Positive Rate - False Positive Rate)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # 5. Calculate Binary Metrics (Precision, Recall, F1)
    # Binarize predictions based on optimal threshold
    y_pred = (y_scores > optimal_threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print Results to Console
    print("\n" + "="*40)
    print(f"🏆 FINAL RESULTS REPORT")
    print("="*40)
    print(f"AUC Score        : {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print("-" * 40)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print("="*40)

    # Save Detailed Metrics to Text File
    with open(os.path.join(OUTPUT_GRAPH_DIR, "results.txt"), "w") as f:
        f.write("EFFICIENT-VAD PRO FINAL RESULTS\n")
        f.write("===============================\n")
        f.write(f"AUC Score        : {roc_auc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.6f}\n\n")
        f.write(f"Accuracy         : {acc:.4f}\n")
        f.write(f"Precision        : {prec:.4f}\n")
        f.write(f"Recall           : {rec:.4f}\n")
        f.write(f"F1-Score         : {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"]))
    
    print(f"[INFO] Detailed metrics saved to '{OUTPUT_GRAPH_DIR}/results.txt'")

    # 6. Plot ROC Curve (enhanced)
    plt.figure(figsize=(8, 7), dpi=300)
    plt.step(fpr, tpr, color='#d62728', lw=3, where='post', label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.fill_between(fpr, tpr, step='post', alpha=0.15, color='#d62728')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7, label='Chance')
    # Annotate optimal threshold
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='black', s=60, zorder=5, label=f'Optimal Threshold\n(FPR={fpr[optimal_idx]:.2f}, TPR={tpr[optimal_idx]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=17, fontweight='bold')
    plt.legend(loc="lower right", fontsize=13, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, "roc_curve.png"), bbox_inches='tight')
    plt.close()
    print("[INFO] Saved ROC Curve.")

    # 7. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    plt.figure(figsize=(8, 8), dpi=300)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold={optimal_threshold:.5f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, "confusion_matrix.png"), bbox_inches='tight')
    print("[INFO] Saved Confusion Matrix.")

    # 7b. Precision-Recall (PR) Curve and Average Precision (AP)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y_true, y_scores)
    # Find precision/recall at optimal threshold
    # thresholds_pr is len N-1, so find closest threshold
    idx_thr = np.argmin(np.abs(thresholds_pr - optimal_threshold)) if len(thresholds_pr) > 0 else -1
    pr_at_thr = precision[idx_thr] if idx_thr >= 0 else None
    rec_at_thr = recall[idx_thr] if idx_thr >= 0 else None
    plt.figure(figsize=(8, 6), dpi=300)
    plt.step(recall, precision, color='purple', lw=2, where='post', label=f'PR curve (AP = {ap:.3f})')
    plt.fill_between(recall, precision, step='post', alpha=0.15, color='purple')
    # Mark optimal threshold point
    if pr_at_thr is not None and rec_at_thr is not None:
        plt.scatter(rec_at_thr, pr_at_thr, color='black', s=60, zorder=5, label=f'@Optimal Threshold\n(P={pr_at_thr:.3f}, R={rec_at_thr:.3f})')
        plt.annotate(f"P={pr_at_thr:.3f}\nR={rec_at_thr:.3f}", (rec_at_thr, pr_at_thr), textcoords="offset points", xytext=(-30,10), ha='right', fontsize=11, color='black', fontweight='bold')
    plt.xlabel('Recall', fontsize=15, fontweight='bold')
    plt.ylabel('Precision', fontsize=15, fontweight='bold')
    plt.title('Precision-Recall (PR) Curve', fontsize=17, fontweight='bold')
    plt.legend(loc='lower left', fontsize=13, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, "pr_curve.png"), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved PR Curve. Average Precision (AP): {ap:.4f}. Precision/Recall at threshold: {pr_at_thr:.4f}/{rec_at_thr:.4f}")

    # 8. Plot Score Histogram
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(y_scores[y_true==0], bins=30, alpha=0.6, color='green', label='Normal Scores')
    plt.hist(y_scores[y_true==1], bins=30, alpha=0.6, color='red', label='Abnormal Scores')
    plt.axvline(optimal_threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('Anomaly Score (MSE)')
    plt.ylabel('Count')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, "score_distribution.png"), bbox_inches='tight')
    print("[INFO] Saved Score Distribution.")

    # 9. t-SNE Visualization (optional, for thesis)
    try:
        # Use features from inference summary if available
        import pandas as pd
        df = pd.read_csv(os.path.join(config.paths.results_dir, "inference", "inference_summary.csv"))
        # For t-SNE, use max_window_error and entropy as features (or extend as needed)
        X = df[["max_window_error", "entropy"]].fillna(0).values
        y = [1 if any(cls in str(vn) for cls in ABNORMAL_CLASSES) else 0 for vn in df["video_name"]]
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6), dpi=150)
        plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='coolwarm', alpha=0.7, label='t-SNE')
        plt.title('t-SNE Visualization of Video Scores')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(label='Class (0=Normal, 1=Abnormal)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_GRAPH_DIR, "tsne_scores.png"), bbox_inches='tight')
        print("[INFO] Saved t-SNE visualization.")
    except Exception as e:
        print(f"[WARN] t-SNE visualization failed: {e}")

    print("\n[SUCCESS] All graphs and metrics generated in './results_graphs_v3'")
    print("\n[SUCCESS] All graphs generated in './results_graphs_v3'")

if __name__ == "__main__":
    # Plot using timeline CSV and label CSV for Explosion008_x264
    timeline_csv = os.path.join(config.paths.results_dir, 'FrameLevel_Results', 'frame_level_labels', 'Explosion008_x264_timeline.csv')
    label_csv = os.path.join(config.paths.results_dir, 'FrameLevel_Results', 'Frame Level Ground Truth', 'Explosion008_x264_labels.csv')
    if os.path.exists(timeline_csv) and os.path.exists(label_csv):
        plot_frame_level_from_timeline(timeline_csv, label_csv, OUTPUT_GRAPH_DIR)
    else:
        print(f"[WARN] Timeline or label CSV not found: {timeline_csv}, {label_csv}")
    efficiency_analysis()
    evaluate()