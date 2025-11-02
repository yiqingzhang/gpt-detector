"""
Batch inference example for GPT Detector.

This script demonstrates how to process multiple texts efficiently
using batch processing.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

from gpt_detector.model import ROBERTAClassifier
from gpt_detector.utils import load_checkpoint


class TextDataset(Dataset):
    """Simple dataset for text classification."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": text
        }


def batch_classify(texts, model_path="saved_models/model.pkl", 
                   config_path="roberta-base", batch_size=8):
    """Classify multiple texts in batches.
    
    Args:
        texts: List of texts to classify.
        model_path: Path to the trained model checkpoint.
        config_path: Path to the RoBERTa configuration.
        batch_size: Number of texts to process at once.
    
    Returns:
        List of predictions ("human generated" or "machine generated").
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(config_path)
    model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3, model_path=config_path)
    model = load_checkpoint(model_path, model)
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process batches
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs, dim=-1).cpu().tolist()
            
            # Convert to labels
            batch_labels = [
                "machine generated" if pred == 0 else "human generated"
                for pred in batch_predictions
            ]
            predictions.extend(batch_labels)
    
    return predictions


def main():
    """Run batch classification example."""
    # Generate example texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming various industries.",
        "I love spending time with my family on weekends.",
        "Machine learning algorithms can process vast amounts of data.",
        "The weather today is absolutely beautiful and sunny.",
        "Deep neural networks have achieved remarkable results.",
        "My favorite hobby is reading books and watching movies.",
        "Natural language processing enables computers to understand text.",
    ]
    
    print("=" * 80)
    print("GPT Detector - Batch Inference Example")
    print("=" * 80)
    print(f"\nProcessing {len(texts)} texts...")
    print()
    
    try:
        predictions = batch_classify(texts, batch_size=4)
        
        print("\nResults:")
        print("-" * 80)
        for i, (text, pred) in enumerate(zip(texts, predictions), 1):
            print(f"{i}. {text[:60]}...")
            print(f"   Prediction: {pred}")
            print()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

