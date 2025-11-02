"""
RoBERTa-based classifier for AI-generated text detection.
"""

import torch.nn as nn
from transformers import RobertaModel


class ROBERTAClassifier(nn.Module):
    """RoBERTa model with classification head for binary text classification.
    
    This model adds a dropout layer and a linear classification layer on top of
    the pre-trained RoBERTa model to classify text as human-generated or AI-generated.
    
    Args:
        n_classes: Number of output classes. Default is 2 (binary classification).
        dropout_rate: Dropout probability for regularization. Default is 0.5.
        model_path: Path or name of the pre-trained RoBERTa model. Default is "roberta-base".
    
    Attributes:
        roberta: Pre-trained RoBERTa model.
        drop: Dropout layer for regularization.
        out: Linear layer for classification.
    
    Example:
        >>> model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3)
        >>> input_ids = torch.randint(0, 1000, (1, 512))
        >>> attention_mask = torch.ones(1, 512)
        >>> output = model(input_ids=input_ids, attention_mask=attention_mask)
    """
    
    def __init__(self, n_classes=2, dropout_rate=0.5, model_path="roberta-base"):
        super(ROBERTAClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.drop = nn.Dropout(dropout_rate)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs from the tokenizer. Shape: (batch_size, seq_length).
            attention_mask: Attention mask for padding tokens. Shape: (batch_size, seq_length).
        
        Returns:
            Logits for each class. Shape: (batch_size, n_classes).
        """
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x1 = self.drop(x.pooler_output)
        x_out = self.out(x1)
        return x_out
