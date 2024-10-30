import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Pandas requires version")
import os
import pandas as pd
from transformers import logging as transformers_logging 
import time
from data_preparation import split_data, save_formatted_data
from openai_integration import upload_training_file, start_fine_tuning, retrieve_fine_tuned_model, generate_exemplar_answer
from evaluation import evaluate_generated_answers, generate_answers_for_test_data

# Define default model ID
DEFAULT_MODEL_ID = 'ft:gpt-4o-mini-2024-07-18:personal::AMkfu4yb'
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_and_upload_data():
    # Step 1: Split data into training, testing, and holdout sets
    input_file = 'cura-llm-training-data.json'
    split_data(input_file)

    # Step 2: Load and format training data for fine-tuning
    train_df = pd.read_json('train_data.json', lines=True)
    save_formatted_data(train_df, 'formatted_train_data.jsonl')

    # Step 3: Upload the formatted training data to OpenAI
    file_response = upload_training_file('formatted_train_data.jsonl')
    print(f"Training file uploaded with ID: {file_response.id}")
    return file_response.id

def run_fine_tuning(file_id):
    job_response = start_fine_tuning(file_id)
    print(f"Fine-tuning started with Job ID: {job_response.id}")
    return job_response.id

def wait_for_fine_tuning_completion(job_id):
    print("Waiting for fine-tuning to complete...")
    model_id = None
    while not model_id:
        time.sleep(30)  # Wait 30 seconds between checks
        model_id = retrieve_fine_tuned_model(job_id)
    print(f"Fine-tuning completed. Fine-tuned model ID: {model_id}")
    return model_id

def retrieve_model_and_generate(model_id):
    # Prompt the user for each part of the input separately
    context = input("Enter the context for the answer generation:\n")
    question = input("Enter the question for the answer generation:\n")
    rubric = input("Enter the rubric for the answer generation:\n")
    
    # Format the prompt using the separate inputs
    prompt = f"Context: {context}\nQuestion: {question}\nRubric: {rubric}"
    
    # Generate the exemplar answer
    answer = generate_exemplar_answer(model_id, prompt)
    print("\n\nGenerated exemplar answer:", answer)

def evaluate_model(model_id):
    # Generate answers for test data using the fine-tuned model
    generate_answers_for_test_data(model_id)
    
    # Evaluate the generated answers using BERTScore
    results = evaluate_generated_answers()
    print("\nBERTScore Evaluation Results:")
    print(results)

def tune_new_model():
    # Step 1: Prepare and upload data for fine-tuning
    file_id = prepare_and_upload_data()
    
    # Step 2: Run fine-tuning process
    job_id = run_fine_tuning(file_id)
    
    # Step 3: Wait for fine-tuning completion and retrieve the model ID
    return wait_for_fine_tuning_completion(job_id)

if __name__ == "__main__":
    print("\nWelcome to the Exemplar Answer Generator!")
    print("This tool is designed to help generate high-quality exemplar answers for educational tasks using a fine-tuned AI model.")
    
    # Initialize fine-tuned model ID if tuning is performed
    fine_tuned_model_id = None
    
    while True:
        print("\nMain Options:")
        print("1. Generate an answer with the fine-tuned model")
        print("2. Start Tuning a new model")
        print("3. Evaluate model")
        print("4. Exit")

        choice = input("Enter the number of your choice: ")

        if choice == '1':
            # Use the fine-tuned model if available; otherwise, use the default model ID
            model_id = fine_tuned_model_id if fine_tuned_model_id else DEFAULT_MODEL_ID
            retrieve_model_and_generate(model_id)

        elif choice == '2':
            # Tune a new model and update the fine-tuned model ID
            fine_tuned_model_id = tune_new_model()

        elif choice == '3':
            # Use the fine-tuned model ID if available for evaluation; otherwise, use the default model ID
            model_id = fine_tuned_model_id if fine_tuned_model_id else DEFAULT_MODEL_ID
            evaluate_model(model_id)

        elif choice == '4':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please enter a number from 1 to 4.")

