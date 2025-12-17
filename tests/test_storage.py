import pytest
from datetime import datetime

def test_minio_connection(minio_client):
    """Test if we can list buckets, implying connection is good"""
    response = minio_client.s3.list_buckets()
    assert 'Buckets' in response
    bucket_names = [b['Name'] for b in response['Buckets']]
    assert 'proofs' in bucket_names
    assert 'retraining-data' in bucket_names

def test_upload_proof(minio_client, dummy_frame):
    """Test uploading a proof image"""
    vehicle_id = 999
    violation_type = "TestViolation"
    
    success = minio_client.save_proof(dummy_frame, vehicle_id, violation_type)
    assert success is True
    
    # Verify file exists
    # Note: We can't easily predict the exact timestamp in the filename, 
    # so we list objects and check if a file with matching prefix exists.
    # In a real test, we might mock the datetime or return the filename from save_proof.
    
    # For now, just checking if upload returned True is a good first step.
    # To be more robust, let's list the bucket and look for recent files.
    
    response = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'])
    assert 'Contents' in response
    found = False
    for obj in response['Contents']:
        if f"{violation_type}_{vehicle_id}" in obj['Key']:
            found = True
            break
    assert found is True

def test_upload_retraining_data(minio_client, dummy_frame, dummy_bbox):
    """Test uploading retraining data (image + label)"""
    vehicle_id = 999
    
    success = minio_client.save_retraining_data(dummy_frame, vehicle_id, dummy_bbox)
    assert success is True
    
    response = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['retraining'])
    assert 'Contents' in response
    
    # Check for image and text file
    found_img = False
    found_txt = False
    for obj in response['Contents']:
        if f"train_{vehicle_id}" in obj['Key']:
            if obj['Key'].endswith('.jpg'):
                found_img = True
            elif obj['Key'].endswith('.txt'):
                found_txt = True
    
    assert found_img is True
    assert found_txt is True

def test_save_labeled_proof(minio_client, dummy_frame, dummy_bbox):
    """Test uploading labeled proof"""
    vehicle_id = 999
    violation_type = "TestViolation"
    
    success = minio_client.save_labeled_proof(dummy_frame, vehicle_id, violation_type, dummy_bbox)
    assert success is True
    
    response = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'])
    assert 'Contents' in response
    found = False
    for obj in response['Contents']:
        if f"{violation_type}_{vehicle_id}" in obj['Key'] and "labeled" in obj['Key']:
            found = True
            break
    assert found is True

def test_save_video_proof(minio_client, dummy_frame):
    """Test uploading video proof"""
    vehicle_id = 999
    violation_type = "TestViolation"
    
    # Create dummy frames as (frame_counter, frame) tuples
    frames = [(i, dummy_frame) for i in range(10)]
    
    success = minio_client.save_video_proof(frames, vehicle_id, violation_type, bboxes=None, fps=10)
    assert success is True
    
    response = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'])
    assert 'Contents' in response
    found = False
    for obj in response['Contents']:
        if f"{violation_type}_{vehicle_id}" in obj['Key'] and obj['Key'].endswith('.mp4'):
            found = True
            break
    assert found is True
