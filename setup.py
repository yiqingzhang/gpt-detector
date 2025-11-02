"""
Setup configuration for GPT Detector package.
"""

import os
from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpt-detector",
    version="0.1.0",
    author="GPT Detector Contributors",
    author_email="",
    description="AI-generated text detection using fine-tuned RoBERTa transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpt-detector",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gpt-detector/issues",
        "Documentation": "https://github.com/yourusername/gpt-detector#readme",
        "Source Code": "https://github.com/yourusername/gpt-detector",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "deployment": [
            "boto3>=1.26.0",
            "sagemaker>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpt-detector=gpt_detector.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
