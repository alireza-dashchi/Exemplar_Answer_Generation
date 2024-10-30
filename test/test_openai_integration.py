import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import MagicMock
from openai_integration import upload_training_file, start_fine_tuning, retrieve_fine_tuned_model, generate_exemplar_answer

# Mock the OpenAI client
@pytest.fixture
def mock_openai_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("openai_integration.client", mock_client)
    return mock_client

# Test for upload_training_file function with nonexistent file
def test_upload_training_file_nonexistent(mock_openai_client):
    mock_openai_client.files.create.side_effect = FileNotFoundError("File not found")
    with pytest.raises(FileNotFoundError):
        upload_training_file("nonexistent_file.jsonl")

# Test for start_fine_tuning function with invalid training file ID
def test_start_fine_tuning_invalid_file_id(mock_openai_client):
    mock_openai_client.fine_tuning.jobs.create.side_effect = ValueError("Invalid file ID")
    with pytest.raises(ValueError):
        start_fine_tuning("invalid_file_id")

# Test for retrieve_fine_tuned_model with nonexistent job ID
def test_retrieve_fine_tuned_model_invalid_job_id(mock_openai_client):
    mock_openai_client.fine_tuning.jobs.retrieve.side_effect = ValueError("Job not found")
    with pytest.raises(ValueError):
        retrieve_fine_tuned_model("nonexistent_job_id")

# Test for generate_exemplar_answer with empty prompt
def test_generate_exemplar_answer_empty_prompt(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = ""
    mock_openai_client.chat.completions.create.return_value = mock_response

    result = generate_exemplar_answer("model-789", "")
    assert result == "", "Expected an empty answer for an empty prompt"

# Test for generate_exemplar_answer with model error
def test_generate_exemplar_answer_model_error(mock_openai_client):
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("Model error")
    with pytest.raises(RuntimeError):
        generate_exemplar_answer("model-789", "sample prompt")
