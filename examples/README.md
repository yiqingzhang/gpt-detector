# Examples

This directory contains example scripts demonstrating how to use GPT Detector.

## Available Examples

### 1. Simple Inference (`simple_inference.py`)

Basic example showing how to classify a single text sample.

```bash
python examples/simple_inference.py
```

**What it demonstrates:**
- Loading a trained model
- Tokenizing input text
- Making predictions
- Interpreting results

### 2. Batch Inference (`batch_inference.py`)

Efficient processing of multiple texts using batch operations.

```bash
python examples/batch_inference.py
```

**What it demonstrates:**
- Creating a custom dataset
- Using DataLoader for batching
- Processing multiple texts efficiently
- Progress tracking with tqdm

### 3. Evaluation (`evaluation.py`)

Complete evaluation example with a trained model.

```bash
python examples/evaluation.py --config src/gpt_detector/config/hyperparameters.json
```

**What it demonstrates:**
- Loading configuration from JSON
- Model evaluation workflow
- Using the complete inference pipeline

## Prerequisites

Before running these examples, ensure you have:

1. Installed the package:
```bash
pip install -e .
```

2. A trained model checkpoint at `saved_models/model.pkl` (or specify your own path)

3. The required dependencies:
```bash
pip install -r requirements.txt
```

## Customization

All examples can be customized by modifying the following parameters:

- `model_path`: Path to your trained model checkpoint
- `config_path`: Path to the RoBERTa configuration or model name
- `batch_size`: Number of samples to process at once (for batch inference)
- `max_length`: Maximum sequence length for tokenization

## Common Issues

### Model Not Found

If you get a "model not found" error, make sure:
- You have trained a model or downloaded a pre-trained checkpoint
- The path to the model is correct
- The model file is not corrupted

### CUDA Out of Memory

If you encounter CUDA memory errors:
- Reduce the batch size
- Use CPU instead: `device = torch.device("cpu")`
- Reduce the maximum sequence length

### Import Errors

If you get import errors:
- Make sure you've installed the package: `pip install -e .`
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify you're using Python 3.8 or higher

## Next Steps

After running these examples, you might want to:

1. Train your own model with custom data
2. Deploy the model to production (see deployment documentation)
3. Integrate the model into your application
4. Fine-tune the model for your specific use case

For more information, see the main [README](../README.md) and [documentation](../docs/).

