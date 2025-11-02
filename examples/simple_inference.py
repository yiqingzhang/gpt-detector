"""
Simple inference example for GPT Detector.

This script demonstrates how to use the trained model to classify
a single text sample as human-generated or AI-generated.
"""

import torch
from transformers import RobertaTokenizer

from gpt_detector.model import ROBERTAClassifier
from gpt_detector.utils import load_checkpoint


def classify_text(text, model_path="saved_models/model.pkl", config_path="roberta-base"):
    """Classify a text sample as human or AI-generated.
    
    Args:
        text: The text to classify.
        model_path: Path to the trained model checkpoint.
        config_path: Path to the RoBERTa configuration.
    
    Returns:
        str: "human generated" or "machine generated"
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config_path)
    
    # Tokenize input
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Load model
    model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3, model_path=config_path)
    model = load_checkpoint(model_path, model)
    model = model.to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs, dim=-1).item()
    
    return "machine generated" if prediction == 0 else "human generated"


def main():
    """Run example classifications."""
    # Example texts
    examples = [
        "The quick brown fox jumps over the lazy dog. This is a simple test sentence.",
        "Artificial intelligence has revolutionized the way we approach complex problems in modern computing.",
        "I went to the store yesterday and bought some groceries. It was a nice sunny day.",
    ]
    
    print("=" * 80)
    print("GPT Detector - Simple Inference Example")
    print("=" * 80)
    print()
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: {text}")
        
        try:
            result = classify_text(text)
            print(f"Prediction: {result}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()

