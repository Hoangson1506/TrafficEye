import pytest
import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.storage import MinioClient

@pytest.fixture(scope="session")
def minio_client():
    client = MinioClient()
    
    # Ensure buckets exist for tests (crucial for CI/CD where MinIO is fresh)
    for bucket_name in client.buckets.values():
        try:
            client.s3.head_bucket(Bucket=bucket_name)
        except Exception:
            try:
                client.s3.create_bucket(Bucket=bucket_name)
            except Exception as e:
                print(f"Warning: Could not create bucket {bucket_name}: {e}")
                
    return client

@pytest.fixture
def dummy_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
    return frame

@pytest.fixture
def dummy_bbox():
    # [x1, y1, x2, y2]
    return [100, 100, 200, 200]
