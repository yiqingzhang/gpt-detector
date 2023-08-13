
# AI-Content-Detector

This repo contains a AI Content Detector.

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest tests
```

## Execution

Training Docker and Pipeline

```bash
build_and_push_train.sh
python ai_detection/pipeline_train.py
```

Results can be viewed on SageMaker Console