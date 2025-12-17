import pytest
from core.vehicle import Vehicle
import numpy as np
from unittest.mock import MagicMock

def test_vehicle_initialization(dummy_bbox):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    assert vehicle.class_id == 1
    assert vehicle.has_violated is False
    assert len(vehicle.violation_type) == 0

def test_vehicle_mark_violation(dummy_bbox, dummy_frame):
    vehicle = Vehicle(dummy_bbox, class_id=1)
    vehicle.id = 123 # Mock ID
    
    # Mock recognizer
    mock_recognizer = MagicMock()
    mock_recognizer.update.return_value = "ABC-123"
    
    # Set threshold to 1 so immediate confirmation happens
    vehicle.vote_threshold = 1
    vehicle.has_violated = True
    
    # Mark violation - passing state (bbox) which is required when frame is provided
    dummy_buffer = [(i, dummy_frame) for i in range(5)]
    state = dummy_bbox  # [x1, y1, x2, y2]
    vehicle.mark_violation("RedLight", mock_recognizer, frame=dummy_frame, frame_buffer=dummy_buffer, fps=10, state=state)
    
    assert vehicle.has_violated is None
    assert "RedLight" in vehicle.violation_type
    assert len(vehicle.proof) > 0
    assert vehicle.proof.shape[2] == 3 # Check channels
    
    # Check if proof is a crop (smaller or equal to frame)
    assert vehicle.proof.size > 0

def test_vehicle_mark_violation_no_frame(dummy_bbox):
    """Test marking violation without a frame (should still update state but no proof)"""
    vehicle = Vehicle(dummy_bbox, class_id=1)
    
    mock_recognizer = MagicMock()
    mock_recognizer.update.return_value = "ABC-123"
    
    vehicle.vote_threshold = 1
    vehicle.has_violated = True
    vehicle.mark_violation("Speeding", mock_recognizer)
    
    assert "Speeding" in vehicle.violation_type
    assert len(vehicle.proof) == 0
