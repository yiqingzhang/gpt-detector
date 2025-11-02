# Contributing to GPT Detector

First off, thank you for considering contributing to GPT Detector! It's people like you that make this tool better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Style Guide](#python-style-guide)
  - [Documentation Style Guide](#documentation-style-guide)
- [Testing](#testing)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Be respectful, constructive, and professional in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, input data, etc.)
- **Describe the behavior you observed** and what you expected to see
- **Include screenshots** if relevant
- **Note your environment**: OS, Python version, package versions

**Bug Report Template:**

```markdown
**Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 20.04, macOS 12.0]
- Python Version: [e.g., 3.8.10]
- Package Version: [e.g., 0.1.0]

**Additional Context:**
Any other relevant information.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed enhancement
- **Explain why this enhancement would be useful**
- **Provide examples** of how it would be used
- **List any alternatives** you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our style guidelines
3. **Add tests** if you've added code that should be tested
4. **Ensure the test suite passes** (`pytest tests/`)
5. **Update documentation** as needed
6. **Write a clear commit message** following our guidelines
7. **Submit your pull request**

**Pull Request Template:**

```markdown
**Description:**
Brief description of changes.

**Related Issue:**
Fixes #(issue number)

**Type of Change:**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Checklist:**
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
```

## Development Setup

### 1. Clone and Install

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/gpt-detector.git
cd gpt-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks (Optional but Recommended)

```bash
pip install pre-commit
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/

# Check code style
black --check src/
flake8 src/
```

## Style Guidelines

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

**Examples:**

```
Add batch processing support for inference

- Implement batch tokenization
- Add batch prediction endpoint
- Update documentation

Fixes #123
```

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: Maximum 100 characters (not 79)
- **Formatting**: Use [Black](https://github.com/psf/black) for automatic formatting
- **Imports**: Organize imports using [isort](https://pycqa.github.io/isort/)
- **Type Hints**: Use type hints for function signatures where appropriate
- **Docstrings**: Use Google-style docstrings

**Example:**

```python
from typing import List, Tuple

import torch
from transformers import RobertaTokenizer


def process_batch(texts: List[str], max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a batch of texts for model input.

    Args:
        texts: List of text strings to process.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A tuple containing:
            - input_ids: Tensor of token IDs
            - attention_mask: Tensor of attention masks

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("texts list cannot be empty")
    
    # Implementation here
    pass
```

### Documentation Style Guide

- Use clear, concise language
- Include code examples for complex features
- Keep examples up-to-date with the codebase
- Use proper Markdown formatting
- Add docstrings to all public functions, classes, and modules

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest tests/ --cov=src/gpt_detector --cov-report=html

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies (API calls, file I/O, etc.)

**Example:**

```python
import pytest
from gpt_detector.model import ROBERTAClassifier


def test_model_initialization():
    """Test that model initializes with correct parameters."""
    model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3)
    assert model is not None
    assert model.drop.p == 0.3


def test_model_forward_pass():
    """Test model forward pass with sample input."""
    model = ROBERTAClassifier(n_classes=2, dropout_rate=0.3)
    input_ids = torch.randint(0, 1000, (1, 512))
    attention_mask = torch.ones(1, 512)
    
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    assert output.shape == (1, 2)
```

## Code Review Process

All submissions require review. We use GitHub pull requests for this purpose:

1. **Automated Checks**: CI/CD will run tests and linting
2. **Peer Review**: At least one maintainer will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, a maintainer will merge your PR

## Recognition

Contributors will be recognized in:
- The project README
- Release notes for significant contributions
- GitHub's contributor graph

## Questions?

Feel free to:
- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Reach out to the maintainers

Thank you for contributing to GPT Detector! 🎉

