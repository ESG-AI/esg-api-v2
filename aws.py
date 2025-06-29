import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
import asyncio

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'ap-southeast-2')
)

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

# Add these utility functions
async def upload_to_s3(file_content, filename=None):
    """Upload content to S3 and return the object key"""
    unique_id = str(uuid.uuid4())
    object_key = f"{unique_id}"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key,
            Body=file_content,
            ContentType='application/pdf'
        )
        return object_key
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None

async def download_from_s3(object_key):
    """Download content from S3 by object key"""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key
        )
        return response['Body'].read()
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return None

async def generate_presigned_url(object_key, expiration=3600):
    """Generate a presigned URL for an object"""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None

async def get_pdf_from_s3(s3_object_key: str) -> bytes:
    """
    Download a PDF file from S3 and return its bytes.
    """
    s3_bucket = os.environ.get("S3_BUCKET_NAME")
    if not s3_bucket:
        raise RuntimeError("S3_BUCKET environment variable not set")
    s3 = boto3.client("s3")
    loop = asyncio.get_event_loop()
    try:
        def download():
            response = s3.get_object(Bucket=s3_bucket, Key=s3_object_key)
            return response["Body"].read()
        file_bytes = await loop.run_in_executor(None, download)
        return file_bytes
    except ClientError as e:
        raise RuntimeError(f"Failed to download {s3_object_key} from S3: {e}")