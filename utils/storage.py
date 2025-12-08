import boto3
from botocore.exceptions import NoCredentialsError
import os
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

class MinioClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MinioClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        load_dotenv()
        self.endpoint_url = "http://localhost:9000"
        self.access_key = "minioadmin"
        self.secret_key = "minioadmin"
        
        # Try to load from env if available
        # In a real app, use python-dotenv or os.environ
        self.access_key = os.environ.get("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.buckets = {
            'proofs': 'proofs',
            'retraining': 'retraining-data',
            'models': 'models'
        }

    def upload_file(self, file_path, bucket_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_path)
        try:
            self.s3.upload_file(file_path, bucket_name, object_name)
            print(f"File {file_path} uploaded to {bucket_name}/{object_name}")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False

    def upload_image_from_memory(self, image_np, bucket_name, object_name):
        """
        Upload a numpy image (OpenCV format) directly to MinIO
        """
        try:
            # Encode image to jpg
            is_success, buffer = cv2.imencode(".jpg", image_np)
            if not is_success:
                print("Failed to encode image")
                return False
            
            io_buf = BytesIO(buffer)
            self.s3.upload_fileobj(io_buf, bucket_name, object_name, ExtraArgs={'ContentType': 'image/jpeg'})
            print(f"Image uploaded to {bucket_name}/{object_name}")
            return True
        except Exception as e:
            print(f"Error uploading image: {e}")
            return False

    def save_proof(self, frame, vehicle_id, violation_type):
        """
        Save violation proof to MinIO
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{vehicle_id}_{timestamp}.jpg"
        return self.upload_image_from_memory(frame, self.buckets['proofs'], filename)

    def save_retraining_data(self, frame, vehicle_id, bbox):
        """
        Save retraining data (full frame + label info)
        For now, just saving the frame. In real scenario, we'd save a corresponding .txt label.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_{vehicle_id}_{timestamp}.jpg"
        
        # Upload image
        if self.upload_image_from_memory(frame, self.buckets['retraining'], filename):
            # Create and upload label (dummy example for YOLO format)
            # class x_center y_center width height
            h, w, _ = frame.shape
            x1, y1, x2, y2 = bbox
            
            # Normalize coordinates
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            label_content = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            label_filename = filename.replace(".jpg", ".txt")
            
            try:
                self.s3.put_object(
                    Bucket=self.buckets['retraining'],
                    Key=label_filename,
                    Body=label_content
                )
                print(f"Label uploaded to {self.buckets['retraining']}/{label_filename}")
                return True
            except Exception as e:
                print(f"Error uploading label: {e}")
                return False
        return False

    def save_labeled_proof(self, frame, vehicle_id, violation_type, bbox):
        """
        Save a full frame with the bounding box drawn on it as proof.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{vehicle_id}_{timestamp}_labeled.jpg"
        
        # Draw bbox
        labeled_frame = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(labeled_frame, f"ID: {vehicle_id} {violation_type}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return self.upload_image_from_memory(labeled_frame, self.buckets['proofs'], filename)

    def save_video_proof(self, frames, vehicle_id, violation_type, fps=30):
        """
        Save a video clip as proof.
        """
        if not frames:
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{vehicle_id}_{timestamp}.mp4"
        
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Upload
            success = self.upload_file(temp_path, self.buckets['proofs'], filename)
            return success
        except Exception as e:
            print(f"Error creating/uploading video proof: {e}")
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
