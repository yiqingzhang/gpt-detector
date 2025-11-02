# Deployment Guide

This guide covers various deployment options for GPT Detector.

## Table of Contents

- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [AWS SageMaker Deployment](#aws-sagemaker-deployment)
- [AWS Lambda Deployment](#aws-lambda-deployment)
- [Production Considerations](#production-considerations)

---

## Local Deployment

### Development Server

For development and testing:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python src/gpt_detector/app.py
```

The server will be available at `http://localhost:5000`.

### Production Server

For production, use a WSGI server like Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 gpt_detector.app:app
```

**Configuration options:**
- `-w 4`: Number of worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
- `--timeout 120`: Request timeout in seconds

---

## Docker Deployment

### Building the Image

#### Inference Container

```bash
cd deployment/docker
docker build -f Dockerfile.inference -t gpt-detector:latest .
```

#### Training Container

```bash
docker build -f Dockerfile.train -t gpt-detector-train:latest .
```

### Running the Container

```bash
# Run inference container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/saved_models:/app/saved_models \
  --name gpt-detector \
  gpt-detector:latest

# Check logs
docker logs gpt-detector

# Stop container
docker stop gpt-detector
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  gpt-detector:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.inference
    ports:
      - "5000:5000"
    volumes:
      - ./saved_models:/app/saved_models
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## AWS SageMaker Deployment

### Prerequisites

1. AWS account with appropriate permissions
2. AWS CLI configured
3. S3 bucket for model artifacts
4. ECR repository for Docker images

### Step 1: Build and Push Training Image

```bash
# Make script executable
chmod +x scripts/build_and_push_train.sh

# Build and push
./scripts/build_and_push_train.sh
```

### Step 2: Run Training Pipeline

```bash
python src/gpt_detector/training/pipeline_train.py
```

This will:
- Upload training data to S3
- Launch a SageMaker training job
- Save the trained model to S3

### Step 3: Build and Push Inference Image

```bash
chmod +x scripts/build_and_push_inference.sh
./scripts/build_and_push_inference.sh
```

### Step 4: Deploy Model

```bash
python src/gpt_detector/training/step_deploy.py
```

### Step 5: Test Endpoint

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = runtime.invoke_endpoint(
    EndpointName='gpt-detector-endpoint',
    ContentType='application/json',
    Body=json.dumps({'inputs': 'Your text to analyze'})
)

result = json.loads(response['Body'].read())
print(result['prediction'])
```

### Monitoring

Monitor your SageMaker endpoint:

```bash
# View endpoint status
aws sagemaker describe-endpoint --endpoint-name gpt-detector-endpoint

# View CloudWatch logs
aws logs tail /aws/sagemaker/Endpoints/gpt-detector-endpoint --follow
```

---

## AWS Lambda Deployment

### Prerequisites

1. AWS account with Lambda permissions
2. ECR repository for Lambda container
3. API Gateway (optional, for HTTP endpoint)

### Step 1: Build Lambda Container

```bash
chmod +x scripts/build_and_push_lambda.sh
./scripts/build_and_push_lambda.sh
```

### Step 2: Deploy with Terraform

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Review changes
terraform plan

# Apply changes
terraform apply
```

### Step 3: Test Lambda Function

```bash
# Using AWS CLI
aws lambda invoke \
  --function-name gpt-detector \
  --payload '{"text": "Your text to analyze"}' \
  response.json

cat response.json
```

### API Gateway Integration

The Terraform configuration includes API Gateway setup. After deployment:

```bash
# Get API endpoint
terraform output api_endpoint

# Test endpoint
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text to analyze"}'
```

---

## Production Considerations

### Security

1. **API Authentication**
   - Implement API keys or OAuth
   - Use AWS IAM for SageMaker/Lambda
   - Enable HTTPS/TLS

2. **Network Security**
   - Use VPC for SageMaker endpoints
   - Configure security groups
   - Enable VPC endpoints for AWS services

3. **Secrets Management**
   - Use AWS Secrets Manager
   - Never commit credentials
   - Rotate credentials regularly

### Scalability

1. **Auto-scaling**
   - Configure SageMaker endpoint auto-scaling
   - Use Lambda concurrency limits
   - Implement request queuing

2. **Load Balancing**
   - Use Application Load Balancer
   - Implement health checks
   - Configure sticky sessions if needed

3. **Caching**
   - Cache frequent predictions
   - Use Redis or ElastiCache
   - Implement CDN for static content

### Monitoring

1. **Metrics**
   - Track request latency
   - Monitor error rates
   - Watch resource utilization

2. **Logging**
   - Centralize logs (CloudWatch, ELK)
   - Implement structured logging
   - Set up log retention policies

3. **Alerting**
   - Configure CloudWatch alarms
   - Set up SNS notifications
   - Create runbooks for incidents

### Cost Optimization

1. **SageMaker**
   - Use appropriate instance types
   - Enable auto-scaling
   - Consider Savings Plans

2. **Lambda**
   - Optimize memory allocation
   - Use provisioned concurrency wisely
   - Monitor cold starts

3. **Storage**
   - Use S3 lifecycle policies
   - Enable S3 Intelligent-Tiering
   - Clean up old artifacts

### Performance

1. **Model Optimization**
   - Use model quantization
   - Enable TorchScript compilation
   - Consider ONNX conversion

2. **Batch Processing**
   - Implement batch inference
   - Use SageMaker Batch Transform
   - Queue requests efficiently

3. **Caching**
   - Cache model in memory
   - Implement prediction caching
   - Use CDN for assets

---

## Troubleshooting

### Common Issues

#### Container Fails to Start

```bash
# Check logs
docker logs gpt-detector

# Common causes:
# - Missing model files
# - Incorrect file paths
# - Port already in use
```

#### SageMaker Training Fails

```bash
# Check CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# Common causes:
# - Insufficient permissions
# - Invalid hyperparameters
# - Data format issues
```

#### Lambda Timeout

```bash
# Increase timeout
aws lambda update-function-configuration \
  --function-name gpt-detector \
  --timeout 300

# Increase memory
aws lambda update-function-configuration \
  --function-name gpt-detector \
  --memory-size 3008
```

---

## Cleanup

### Docker

```bash
docker stop gpt-detector
docker rm gpt-detector
docker rmi gpt-detector:latest
```

### AWS Resources

```bash
# Using Terraform
cd deployment/terraform
terraform destroy

# Manual cleanup
aws sagemaker delete-endpoint --endpoint-name gpt-detector-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name gpt-detector-config
aws sagemaker delete-model --model-name gpt-detector-model
```

---

## Support

For deployment issues:
- Check [GitHub Issues](https://github.com/yourusername/gpt-detector/issues)
- Review [API Documentation](API.md)
- See [Contributing Guidelines](../CONTRIBUTING.md)

