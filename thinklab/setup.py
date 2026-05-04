import re
from pathlib import Path
from setuptools import setup, find_packages

# Read version from thinklab/__init__.py (single source of truth)
init_file = Path(__file__).parent / "thinklab" / "__init__.py"
version = re.search(r'__version__\s*=\s*"(.+?)"', init_file.read_text()).group(1)

setup(
    name="thinklab",
    version=version,
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
    author="Abel Yohannes",
    description="Pure PyTorch multimodal AI framework with runtime explainability",
    url="https://github.com/abeldirectory252/ThinkAi",
)
