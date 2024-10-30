import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from main import prepare_and_upload_data, run_fine_tuning, wait_for_fine_tuning_completion
from unittest.mock import MagicMock

# Mock dependencies and run full flow in sequence
def test_full_process_flow(monkeypatch):
    # Mock each function in the process flow
    mock_upload = MagicMock(return_value=MagicMock(id="file-123"))
    mock_start = MagicMock(return_value=MagicMock(id="job-456"))
    mock_retrieve = MagicMock(side_effect=[None, "model-789"])
    
    monkeypatch.setattr("main.upload_training_file", mock_upload)
    monkeypatch.setattr("main.start_fine_tuning", mock_start)
    monkeypatch.setattr("main.retrieve_fine_tuned_model", mock_retrieve)
    
    # Step 1: Prepare and upload data
    file_id = prepare_and_upload_data()
    assert file_id == "file-123"
    
    # Step 2: Start fine-tuning process
    job_id = run_fine_tuning(file_id)
    assert job_id == "job-456"
    
    # Step 3: Wait for fine-tuning completion and retrieve model ID
    model_id = wait_for_fine_tuning_completion(job_id)
    assert model_id == "model-789"
    
# Test for prepare_and_upload_data with invalid file path
def test_prepare_and_upload_data_invalid_file(monkeypatch):
    # Mock the upload function to simulate failure on invalid file
    mock_upload = MagicMock(side_effect=FileNotFoundError("File not found"))
    monkeypatch.setattr("main.upload_training_file", mock_upload)

    with pytest.raises(FileNotFoundError):
        prepare_and_upload_data()