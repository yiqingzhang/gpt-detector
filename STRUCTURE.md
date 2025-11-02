# Project Structure

This document provides a visual overview of the GPT Detector project structure.

## Directory Tree

```
gpt-detector/
│
├── .github/                          # GitHub configuration
│   └── workflows/                    # CI/CD workflows
│       ├── ci.yml                    # Continuous Integration
│       └── publish.yml               # PyPI publishing
│
├── src/                              # Source code (main package)
│   └── gpt_detector/                 # Main package
│       ├── __init__.py               # Package initialization
│       ├── model.py                  # RoBERTa classifier model
│       ├── utils.py                  # Utility functions
│       ├── app.py                    # Flask web application
│       ├── data_process.py           # Data preprocessing
│       ├── config/                   # Configuration files
│       │   └── hyperparameters.json  # Training hyperparameters
│       └── training/                 # Training modules
│           ├── __init__.py           # Training package init
│           ├── train.py              # Training script
│           ├── inference.py          # SageMaker inference
│           ├── pipeline_train.py     # SageMaker pipeline
│           └── step_deploy.py        # Deployment script
│
├── examples/                         # Usage examples
│   ├── README.md                     # Examples documentation
│   ├── simple_inference.py           # Basic inference example
│   ├── batch_inference.py            # Batch processing example
│   └── evaluation.py                 # Evaluation example
│
├── docs/                             # Documentation
│   ├── README.md                     # Documentation hub
│   ├── API.md                        # API reference
│   ├── DEPLOYMENT.md                 # Deployment guide
│   └── postman/                      # API testing
│       └── query.postman_collection.json
│
├── deployment/                       # Deployment configurations
│   ├── docker/                       # Docker files
│   │   ├── Dockerfile.inference      # Inference container
│   │   ├── Dockerfile.train          # Training container
│   │   └── Dockerfile.lambda         # Lambda container
│   ├── terraform/                    # Infrastructure as Code
│   │   ├── main.tf                   # Main Terraform config
│   │   ├── variables.tf              # Variables
│   │   ├── providers.tf              # Cloud providers
│   │   ├── network.tf                # Network config
│   │   ├── storage.tf                # Storage config
│   │   └── local.tf                  # Local config
│   ├── handler.py                    # Lambda handler
│   └── lambda-sagemaker-api.py       # Lambda-SageMaker API
│
├── scripts/                          # Build and deployment scripts
│   ├── build_and_push_train.sh       # Build training image
│   ├── build_and_push_inference.sh   # Build inference image
│   ├── build_and_push_lambda.sh      # Build Lambda image
│   └── build_model.sh                # Build model script
│
├── tests/                            # Test suite
│   ├── conftest.py                   # Test configuration
│   ├── test_model.py                 # Model tests
│   ├── test_utils.py                 # Utility tests
│   ├── test_dataprocess.py           # Data processing tests
│   └── test_train_roberta.py         # Training tests
│
├── data/                             # Training/test data
│   ├── train/                        # Training data
│   ├── test/                         # Test data
│   └── val/                          # Validation data
│
├── saved_models/                     # Model checkpoints
│   ├── config.json                   # Model configuration
│   ├── vocab.json                    # Vocabulary
│   ├── merges.txt                    # BPE merges
│   ├── dict.txt                      # Dictionary
│   └── metric.pkl                    # Training metrics
│
├── templates/                        # HTML templates
│   └── my_template.html              # Web UI template
│
├── .gitignore                        # Git ignore rules
├── .pre-commit-config.yaml           # Pre-commit hooks
├── pyproject.toml                    # Project configuration
├── setup.py                          # Package setup
│
├── requirements.txt                  # Core dependencies
├── requirements-dev.txt              # Development dependencies
├── requirements-deploy.txt           # Deployment dependencies
│
├── README.md                         # Project overview
├── QUICKSTART.md                     # Quick start guide
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License
├── CHANGELOG.md                      # Version history
├── PROJECT_STATUS.md                 # Project status
├── REFACTORING_SUMMARY.md            # Refactoring details
└── STRUCTURE.md                      # This file
```

## Key Directories

### `/src/gpt_detector/`
**Purpose:** Main application code

**Contains:**
- Core model implementation
- Utility functions
- Web application
- Data processing
- Training modules

**Key Files:**
- `model.py` - RoBERTa classifier
- `app.py` - Flask web server
- `utils.py` - Helper functions
- `data_process.py` - Data preprocessing

---

### `/examples/`
**Purpose:** Usage examples and tutorials

**Contains:**
- Simple inference example
- Batch processing example
- Evaluation example
- Examples documentation

**Use Case:** Learning how to use the package

---

### `/docs/`
**Purpose:** Project documentation

**Contains:**
- API reference
- Deployment guides
- Architecture documentation
- Postman collections

**Use Case:** Understanding and integrating the project

---

### `/deployment/`
**Purpose:** Deployment configurations

**Contains:**
- Docker files for different environments
- Terraform infrastructure code
- Lambda handlers
- Deployment scripts

**Use Case:** Deploying to production

---

### `/scripts/`
**Purpose:** Build and automation scripts

**Contains:**
- Docker build scripts
- Model building scripts
- Deployment automation

**Use Case:** Building and deploying the application

---

### `/tests/`
**Purpose:** Test suite

**Contains:**
- Unit tests
- Integration tests
- Test fixtures

**Use Case:** Ensuring code quality

---

## File Purposes

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Modern Python project configuration |
| `setup.py` | Package installation configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |
| `.gitignore` | Git ignore rules |
| `requirements*.txt` | Dependency specifications |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `QUICKSTART.md` | 5-minute getting started guide |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history |
| `LICENSE` | MIT License |
| `PROJECT_STATUS.md` | Current project status |
| `REFACTORING_SUMMARY.md` | Refactoring details |

### CI/CD Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Automated testing and linting |
| `.github/workflows/publish.yml` | PyPI publishing automation |

## Import Structure

```python
# Top-level package
from gpt_detector import ROBERTAClassifier, load_checkpoint, save_checkpoint

# Model module
from gpt_detector.model import ROBERTAClassifier

# Utils module
from gpt_detector.utils import load_checkpoint, save_checkpoint, parse_args

# Data processing
from gpt_detector.data_process import run_data_preprocess

# Training
from gpt_detector.training import train, validation
```

## Entry Points

### Command Line
```bash
# Web application
python src/gpt_detector/app.py

# Data preprocessing
python src/gpt_detector/data_process.py

# Training
python src/gpt_detector/training/train.py

# Examples
python examples/simple_inference.py
python examples/batch_inference.py
python examples/evaluation.py
```

### Python API
```python
# Import and use
from gpt_detector import ROBERTAClassifier
model = ROBERTAClassifier()
```

### Web API
```bash
# REST endpoints
GET  /health
POST /
POST /predict
```

## Development Workflow

```
1. Clone repository
   ↓
2. Install dependencies (requirements-dev.txt)
   ↓
3. Install package (pip install -e .)
   ↓
4. Set up pre-commit hooks
   ↓
5. Make changes
   ↓
6. Run tests (pytest)
   ↓
7. Format code (black, isort)
   ↓
8. Commit changes
   ↓
9. Push and create PR
```

## Deployment Workflow

```
1. Choose deployment option
   ├── Local (Flask)
   ├── Docker
   ├── AWS SageMaker
   └── AWS Lambda
   ↓
2. Build artifacts
   ├── Docker images
   └── Model checkpoints
   ↓
3. Configure infrastructure
   ├── Terraform
   └── AWS resources
   ↓
4. Deploy
   ↓
5. Monitor and maintain
```

## Data Flow

```
Input Text
    ↓
Tokenizer (RoBERTa)
    ↓
Model (ROBERTAClassifier)
    ├── RoBERTa Base
    ├── Dropout
    └── Linear Layer
    ↓
Prediction (Human/AI)
```

## Summary

- **Total Directories:** 15+
- **Total Files:** 50+
- **Python Modules:** 12
- **Documentation Files:** 8
- **Example Scripts:** 3
- **Test Files:** 5
- **Configuration Files:** 6

---

**This structure follows Python best practices and is designed for:**
- Easy navigation
- Clear separation of concerns
- Scalability
- Maintainability
- Contributor-friendliness

---

Last updated: November 2, 2025

