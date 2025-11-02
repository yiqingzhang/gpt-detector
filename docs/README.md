# Documentation

Welcome to the GPT Detector documentation!

## Quick Links

- [API Documentation](API.md) - Complete API reference
- [Deployment Guide](DEPLOYMENT.md) - Deployment instructions for various platforms
- [Examples](../examples/README.md) - Code examples and tutorials
- [Contributing](../CONTRIBUTING.md) - How to contribute to the project

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt-detector.git
cd gpt-detector

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Quick Start

```python
from gpt_detector.model import ROBERTAClassifier
from gpt_detector.utils import load_checkpoint
import torch

# Load model
model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3)
model = load_checkpoint("saved_models/model.pkl", model)

# Make prediction
# ... (see examples for complete code)
```

## Documentation Structure

### For Users

- **[README](../README.md)** - Project overview and quick start
- **[Examples](../examples/)** - Practical examples
- **[API Documentation](API.md)** - Complete API reference

### For Developers

- **[Contributing Guide](../CONTRIBUTING.md)** - Development guidelines
- **[Deployment Guide](DEPLOYMENT.md)** - Deployment instructions
- **Code Documentation** - Inline docstrings in source code

### For DevOps

- **[Deployment Guide](DEPLOYMENT.md)** - Infrastructure setup
- **Docker Files** - Container configurations in `deployment/docker/`
- **Terraform** - IaC configurations in `deployment/terraform/`

## Architecture Overview

```
┌─────────────────┐
│   User Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenizer     │ (RoBERTa)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RoBERTa Model  │ (Pre-trained)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │ (Dropout + Linear)
│     Head        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │ (Human/AI)
└─────────────────┘
```

## Key Components

### Model (`src/gpt_detector/model.py`)

The core classification model built on RoBERTa:
- Pre-trained RoBERTa base
- Dropout layer for regularization
- Linear classification head

### Training (`src/gpt_detector/training/`)

Complete training pipeline:
- Data preprocessing
- Model training with validation
- Checkpoint management
- SageMaker integration

### Inference (`src/gpt_detector/app.py`)

Flask-based web API:
- REST endpoints
- Form and JSON input
- Error handling
- Health checks

## Common Tasks

### Training a Model

```bash
# Prepare data
python src/gpt_detector/data_process.py

# Train locally
python src/gpt_detector/training/train.py

# Or train on SageMaker
python src/gpt_detector/training/pipeline_train.py
```

### Running Inference

```bash
# Start web server
python src/gpt_detector/app.py

# Or use Python API
python examples/simple_inference.py
```

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/gpt_detector
```

## Configuration

Configuration is managed through JSON files in `src/gpt_detector/config/`:

```json
{
    "saved_model_dir": "/opt/ml/model",
    "datafolder": "data",
    "output_path": "/opt/ml/model",
    "epochs": 10,
    "train_batch_size": 16,
    "lr": 5e-5,
    "seed": 42,
    "dropout_rate": 0.3
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `production` |
| `FLASK_APP` | Flask app module | `gpt_detector.app` |
| `MODEL_PATH` | Path to model checkpoint | `saved_models/model.pkl` |

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure package is installed
pip install -e .
```

**Model Not Found**
```bash
# Check model path in config
# Ensure model file exists
ls -la saved_models/
```

**CUDA Out of Memory**
```python
# Use CPU instead
device = torch.device("cpu")

# Or reduce batch size
batch_size = 4
```

## Performance Tips

1. **Use GPU** - Significantly faster inference
2. **Batch Processing** - Process multiple texts together
3. **Model Caching** - Keep model in memory
4. **Optimize Sequence Length** - Use appropriate max_length

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/gpt-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gpt-detector/discussions)
- **Email**: [Contact maintainers]

## Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

Last updated: November 2025

