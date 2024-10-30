import json
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation import evaluate_generated_answers, generate_answers_for_test_data

def split_data(file_path, train_output='train_data.json', test_output='test_data.json', holdout_output='holdout_data.json', test_size=0.3, holdout_size=0.3):
    # Load the original data
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Check if the DataFrame is empty
    if df.empty:
        # Save empty DataFrames to each output file
        pd.DataFrame().to_json(train_output, orient='records', lines=True)
        pd.DataFrame().to_json(test_output, orient='records', lines=True)
        pd.DataFrame().to_json(holdout_output, orient='records', lines=True)
        print("Input data is empty. Created empty train, test, and holdout files.")
        return

    # Split data into 70% for training and 30% for testing
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=100)

    # Further split the test set into 70% for testing and 30% for holdout
    test_df, holdout_df = train_test_split(test_df, test_size=holdout_size, random_state=100)

    # Save split data
    train_df.to_json(train_output, orient='records', lines=True)
    test_df.to_json(test_output, orient='records', lines=True)
    holdout_df.to_json(holdout_output, orient='records', lines=True)
    print(f"Data split completed. Training data saved to {train_output}, testing data saved to {test_output}, and holdout data saved to {holdout_output}.")

def format_training_example(task_content, question, rubric, answer):
    return {
        "messages": [
            {"role": "system", "content": "You are an expert educator."},
            {"role": "user", "content": f"Context: {task_content}\n\nQuestion: {question}\n\nRubric: {rubric}"},
            {"role": "assistant", "content": answer.strip('"')}
        ]
    }

def save_formatted_data(df, output_file='formatted_train_data.jsonl'):
    formatted_data = [
        format_training_example(row['task_content'], row['question'], row['rubric'], row['answer'])
        for _, row in df.iterrows()
    ]
    with open(output_file, 'w') as file:
        for entry in formatted_data:
            json.dump(entry, file)
            file.write('\n')
    print(f"Data formatted and saved to {output_file}.")

if __name__ == "__main__":
    # File path to the original data
    input_file = 'cura-llm-training-data.json'
    
    # Step 1: Split data into training, testing, and holdout sets
    split_data(input_file)

    # Step 2: Load the training data and format it for fine-tuning
    train_df = pd.read_json('train_data.json', lines=True)
    save_formatted_data(train_df, output_file='formatted_train_data.jsonl')
