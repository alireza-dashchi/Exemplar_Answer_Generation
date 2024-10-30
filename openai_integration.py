import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client using 
client = OpenAI(api_key=api_key)

def upload_training_file(file_path):
    upload_response = client.files.create(
    file=open("formatted_train_data.jsonl", "rb"),
    purpose="fine-tune"
    )
    return upload_response

def start_fine_tuning(training_file_id, model="gpt-4o-mini-2024-07-18"):
    fine_tune_response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model=model
    ) 
    return fine_tune_response

def retrieve_fine_tuned_model(job_id):
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job.fine_tuned_model

def generate_exemplar_answer(model_id, prompt):
    system_instructions = "You are an expert educator creating high-quality exemplar answers based on rubrics and educational context."

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise 

