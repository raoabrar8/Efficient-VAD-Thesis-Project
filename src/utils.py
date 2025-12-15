import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_gaussian_smoothing(scores, sigma=15):
    """
    Step 5: Smoothing
    Applies Gaussian smoothing to the error scores to reduce noise.
    
    Args:
        scores (list or np.array): Raw anomaly scores (MSE per frame).
        sigma (int): The smoothing factor. Higher = smoother.
        
    Returns:
        np.array: Smoothed scores.
    """
    return gaussian_filter1d(scores, sigma=sigma)

def get_threshold(scores):
    """
    Step 5: Decision
    Calculates the anomaly threshold based on the stats of the scores.
    Formula: Threshold = Mean + 3 * Std_Dev
    
    Note: In a real production system, this should be calculated on the 
    Validation Set (Normal videos only), not the Test video itself.
    For this thesis demo, we calculate it dynamically or use a fixed value.
    """
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + (3 * std_score)
    return threshold

def min_max_norm(scores):
    """
    Optional: Normalize scores to 0-1 range for better visualization.
    """
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())