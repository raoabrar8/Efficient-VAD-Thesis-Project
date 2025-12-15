"""
Setup script for Efficient-VAD Thesis Project
"""

from setuptools import setup, find_packages

setup(
    name="efficient-vad",
    version="1.0.0",
    description="Efficient Anomaly Detection in Surveillance Videos using Pretrained CNN and Self-Supervised Learning",
    author="Thesis Project",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
)



