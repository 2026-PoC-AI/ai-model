# app/core/s3.py
import json
import boto3
from app.core.config import settings
from botocore.exceptions import ClientError
from typing import Union
from app.core.config import settings


class S3Client:
    def __init__(self):
        import logging
        logger = logging.getLogger("app")
        
        # 설정값 디버깅
        logger.info(f"=== S3 Client Configuration ===")
        logger.info(f"Bucket: {settings.AWS_BUCKET_NAME}")
        logger.info(f"Region: {settings.AWS_REGION}")
        logger.info(f"Access Key: {settings.AWS_ACCESS_KEY_ID[:10]}...{settings.AWS_ACCESS_KEY_ID[-4:]}")
        logger.info(f"Secret Key: {settings.AWS_SECRET_ACCESS_KEY[:10]}...{settings.AWS_SECRET_ACCESS_KEY[-4:]}")
        logger.info(f"================================")
        
        self.bucket = settings.AWS_BUCKET_NAME
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

    # =========================
    # Upload (bytes / file)
    # =========================
    def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def upload_file(
        self,
        key: str,
        file_path: str,
        content_type: str,
    ) -> None:
        self.s3.upload_file(
            Filename=file_path,
            Bucket=self.bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )

    # =========================
    # JSON helpers
    # =========================
    def upload_json(
        self,
        key: str,
        payload: dict,
    ) -> None:
        self.upload_bytes(
            key=key,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
        )

    def download_bytes(self, key: str) -> bytes:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            raise e
        
    def download_bytes(self, key: str) -> bytes:
        import logging
        logger = logging.getLogger("app")
        
        try:
            logger.info(f"Downloading from S3: bucket={self.bucket}, key={key}")
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = response["Body"].read()
            logger.info(f"Successfully downloaded {len(data)} bytes")
            return data
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"S3 download failed - Code: {error_code}, Message: {error_message}")
            logger.error(f"Bucket: {self.bucket}, Key: {key}")
            raise e

# 싱글톤으로 사용
s3_client = S3Client()

# 즉시 설정 확인
import logging
logger = logging.getLogger("app")
logger.info(f"=== S3 Client Singleton Created ===")
logger.info(f"Bucket: {s3_client.bucket}")
logger.info(f"Region: {settings.AWS_REGION}")
logger.info(f"Access Key ID: {settings.AWS_ACCESS_KEY_ID}")
logger.info(f"Secret Access Key length: {len(settings.AWS_SECRET_ACCESS_KEY) if settings.AWS_SECRET_ACCESS_KEY else 0}")
