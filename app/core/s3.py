# app/core/s3.py
import json
import boto3
from typing import Union
from app.core.config import settings


class S3Client:
    def __init__(self):
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

# 싱글톤으로 써도 OK
s3_client = S3Client()
