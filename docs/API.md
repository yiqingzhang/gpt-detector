# API Documentation

## Flask Web API

The GPT Detector provides a Flask-based REST API for easy integration into web applications.

### Starting the Server

```bash
python src/gpt_detector/app.py
```

The server will start on `http://localhost:5000` by default.

### Endpoints

#### Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:5000/health
```

---

#### Predict (Form Data)

Classify text using form data submission.

**Endpoint:** `POST /`

**Content-Type:** `application/x-www-form-urlencoded`

**Parameters:**
- `text` (string, required): The text to classify

**Response:** Plain text string
- `"machine generated"` or `"human generated"`

**Example:**
```bash
curl -X POST http://localhost:5000/ \
  -F "text=Your text to analyze here"
```

---

#### Predict (JSON)

Classify text using JSON payload.

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "text": "Your text to analyze here"
}
```

**Response:**
```json
{
  "message": "Success",
  "input_data": "human generated"
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "error_msg": "Detailed error traceback"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze here"}'
```

---

## Python API

### Model Class

#### `ROBERTAClassifier`

RoBERTa-based classifier for binary text classification.

```python
from gpt_detector.model import ROBERTAClassifier

model = ROBERTAClassifier(
    n_classes=2,
    dropout_rate=0.3,
    model_path="roberta-base"
)
```

**Parameters:**
- `n_classes` (int): Number of output classes. Default: 2
- `dropout_rate` (float): Dropout probability. Default: 0.5
- `model_path` (str): Path to pre-trained RoBERTa model. Default: "roberta-base"

**Methods:**

##### `forward(input_ids, attention_mask)`

Perform forward pass through the model.

**Parameters:**
- `input_ids` (torch.Tensor): Token IDs. Shape: (batch_size, seq_length)
- `attention_mask` (torch.Tensor): Attention mask. Shape: (batch_size, seq_length)

**Returns:**
- `torch.Tensor`: Logits for each class. Shape: (batch_size, n_classes)

---

### Utility Functions

#### `load_checkpoint(path, model)`

Load model weights from a checkpoint file.

```python
from gpt_detector.utils import load_checkpoint

model = load_checkpoint("path/to/model.pkl", model)
```

**Parameters:**
- `path` (str): Path to checkpoint file
- `model` (nn.Module): Model instance to load weights into

**Returns:**
- `nn.Module`: Model with loaded weights

---

#### `save_checkpoint(path, model)`

Save model weights to a checkpoint file.

```python
from gpt_detector.utils import save_checkpoint

save_checkpoint("path/to/model.pkl", model)
```

**Parameters:**
- `path` (str): Path where checkpoint will be saved
- `model` (nn.Module): Model to save

---

#### `parse_arge()`

Parse configuration from JSON file.

```python
from gpt_detector.utils import parse_arge

args = parse_arge()
print(args.epochs)  # Access configuration values
```

**Returns:**
- `argparse.Namespace`: Configuration object

---

### Data Processing

#### `run_data_preprocess(args)`

Execute the complete data preprocessing pipeline.

```python
from gpt_detector.data_process import run_data_preprocess
from gpt_detector.utils import parse_arge

args = parse_arge()
run_data_preprocess(args)
```

**Parameters:**
- `args` (argparse.Namespace): Configuration arguments

**Pipeline Steps:**
1. Load dataset from Hugging Face
2. Transform labels to binary classification
3. Create smaller subsets (optional)
4. Tokenize text data
5. Save processed datasets

---

### Training

#### `train(args, model, device, dataloader_train, dataloader_val, optimizer, criterion, scheduler, validation_mode, checkpoint, metrics, save_dir)`

Train the model.

```python
from gpt_detector.training.train import train

train(
    args=args,
    model=model,
    device=device,
    dataloader_train=train_loader,
    dataloader_val=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    validation_mode=True,
    checkpoint=True,
    metrics=True,
    save_dir="/path/to/save"
)
```

**Parameters:**
- `args`: Configuration arguments
- `model`: Model to train
- `device`: PyTorch device (CPU/GPU)
- `dataloader_train`: Training data loader
- `dataloader_val`: Validation data loader
- `optimizer`: PyTorch optimizer
- `criterion`: Loss function
- `scheduler`: Learning rate scheduler (optional)
- `validation_mode` (bool): Whether to run validation
- `checkpoint` (bool): Whether to save checkpoints
- `metrics` (bool): Whether to save metrics
- `save_dir` (str): Directory to save outputs

---

## AWS SageMaker API

### Inference Handler

The SageMaker inference handler provides the following functions:

#### `model_fn(model_dir)`

Load the model from the specified directory.

#### `input_fn(request_body, request_content_type)`

Deserialize the request body.

#### `predict_fn(data, model_and_tokenizer)`

Make predictions using the loaded model.

#### `output_fn(prediction, response_content_type)`

Serialize the prediction results.

### Example SageMaker Request

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='gpt-detector-endpoint',
    ContentType='application/json',
    Body=json.dumps({'inputs': 'Your text to analyze'})
)

result = json.loads(response['Body'].read())
print(result['prediction'])
```

---

## Error Handling

All API endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input or processing error
- `500 Internal Server Error`: Server-side error

Error responses include:
```json
{
  "error": "Brief error description",
  "error_msg": "Detailed error message"
}
```

---

## Rate Limiting

Currently, there are no rate limits on the API. For production deployments, consider implementing rate limiting using:

- Flask-Limiter
- API Gateway (for AWS deployments)
- Nginx rate limiting

---

## Authentication

The current implementation does not include authentication. For production use, consider adding:

- API keys
- OAuth 2.0
- JWT tokens
- AWS IAM authentication (for SageMaker endpoints)

---

## Best Practices

1. **Input Validation**: Always validate text input before sending to the API
2. **Error Handling**: Implement proper error handling in your client code
3. **Timeouts**: Set appropriate timeouts for API requests
4. **Batch Processing**: Use batch inference for processing multiple texts
5. **Caching**: Cache predictions for frequently analyzed texts
6. **Monitoring**: Monitor API performance and error rates

---

## Support

For issues or questions:
- GitHub Issues: [Report a bug](https://github.com/yourusername/gpt-detector/issues)
- Documentation: [Main README](../README.md)
- Examples: [Example scripts](../examples/)

