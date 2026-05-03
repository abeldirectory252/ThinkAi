from setuptools import setup, find_packages

setup(
    name="thinklab",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    author="ThinkLab",
    description="Pure PyTorch AI research framework with model explainability",
)
