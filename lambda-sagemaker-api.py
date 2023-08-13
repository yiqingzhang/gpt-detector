import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    payload = json.loads(json.dumps(event))
    
    if "text" not in payload:
        payload = json.loads(payload['body'])
        assert "text" in payload
    
    print(payload)
    
    payload_bytes = json.dumps(payload).encode('utf-8')
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload_bytes)
    print(response)
    
    result = json.loads(response['Body'].read().decode())
    
    print(result)
    return result