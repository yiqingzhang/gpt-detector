# Quick Start Guide

Get up and running with GPT Detector in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU

## Installation

### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt-detector.git
cd gpt-detector

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Development Installation

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/gpt-detector.git
cd gpt-detector
pip install -r requirements-dev.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### Option 3: Docker Installation

```bash
# Pull and run the Docker image
docker pull yourusername/gpt-detector:latest
docker run -p 5000:5000 gpt-detector:latest
```

## First Steps

### 1. Verify Installation

```python
# Test import
python -c "from gpt_detector import ROBERTAClassifier; print('✓ Installation successful!')"
```

### 2. Download or Train a Model

**Option A: Use Pre-trained Model** (if available)
```bash
# Download model checkpoint
# Place in saved_models/model.pkl
```

**Option B: Train Your Own Model**
```bash
# Prepare data
python src/gpt_detector/data_process.py

# Train model
python src/gpt_detector/training/train.py
```

### 3. Run Your First Prediction

```python
import torch
from transformers import RobertaTokenizer
from gpt_detector.model import ROBERTAClassifier
from gpt_detector.utils import load_checkpoint

# Load model
model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3)
model = load_checkpoint("saved_models/model.pkl", model)
model.eval()

# Prepare input
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
text = "Your text to analyze here"
encoded = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

# Predict
with torch.no_grad():
    output = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
    prediction = torch.argmax(output, dim=-1).item()

result = "machine generated" if prediction == 0 else "human generated"
print(f"Prediction: {result}")
```

### 4. Start the Web Server

```bash
# Start Flask application
python src/gpt_detector/app.py

# Server runs on http://localhost:5000
```

Test the API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze"}'
```

## Common Use Cases

### Use Case 1: Single Text Classification

```python
from examples.simple_inference import classify_text

result = classify_text("Your text here")
print(result)  # "human generated" or "machine generated"
```

### Use Case 2: Batch Processing

```python
from examples.batch_inference import batch_classify

texts = [
    "Text 1 to analyze",
    "Text 2 to analyze",
    "Text 3 to analyze"
]

results = batch_classify(texts, batch_size=8)
for text, result in zip(texts, results):
    print(f"{text[:50]}... → {result}")
```

### Use Case 3: Web API Integration

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "Your text to analyze"}
)

result = response.json()
print(result["input_data"])  # "human generated" or "machine generated"
```

## Configuration

Edit `src/gpt_detector/config/hyperparameters.json`:

```json
{
    "saved_model_dir": "roberta-base",
    "datafolder": "data",
    "output_path": "saved_models",
    "epochs": 10,
    "train_batch_size": 16,
    "lr": 5e-5,
    "seed": 42,
    "dropout_rate": 0.3
}
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/gpt_detector

# Run specific test
pytest tests/test_model.py -v
```

## Troubleshooting

### Issue: Import Error

**Problem:** `ModuleNotFoundError: No module named 'gpt_detector'`

**Solution:**
```bash
pip install -e .
```

### Issue: Model Not Found

**Problem:** `FileNotFoundError: model.pkl not found`

**Solution:**
- Train a model: `python src/gpt_detector/training/train.py`
- Or download a pre-trained checkpoint
- Ensure the path in config is correct

### Issue: CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Use CPU instead
device = torch.device("cpu")

# Or reduce batch size
batch_size = 4
```

### Issue: Flask Port Already in Use

**Problem:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Use a different port
export FLASK_RUN_PORT=5001
python src/gpt_detector/app.py
```

## Next Steps

1. **Read the Documentation**
   - [Full README](README.md)
   - [API Documentation](docs/API.md)
   - [Deployment Guide](docs/DEPLOYMENT.md)

2. **Try the Examples**
   - [Simple Inference](examples/simple_inference.py)
   - [Batch Processing](examples/batch_inference.py)
   - [Evaluation](examples/evaluation.py)

3. **Deploy to Production**
   - [Docker Deployment](docs/DEPLOYMENT.md#docker-deployment)
   - [AWS SageMaker](docs/DEPLOYMENT.md#aws-sagemaker-deployment)
   - [AWS Lambda](docs/DEPLOYMENT.md#aws-lambda-deployment)

4. **Contribute**
   - Read [CONTRIBUTING.md](CONTRIBUTING.md)
   - Check [open issues](https://github.com/yourusername/gpt-detector/issues)
   - Submit a pull request

## Useful Commands

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type check
mypy src/

# Run all quality checks
pre-commit run --all-files

# Build package
python -m build

# Install from local build
pip install dist/gpt_detector-0.1.0-py3-none-any.whl
```

## Getting Help

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/gpt-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gpt-detector/discussions)

## Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Ready to detect AI-generated text?** Start with the examples and explore the API! 🚀

