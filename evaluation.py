import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Pandas requires version")
import pandas as pd
import json
from openai_integration import generate_exemplar_answer
from bert_score import score
from transformers import logging as transformers_logging 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set transformers logging level to ERROR to suppress warnings
transformers_logging.set_verbosity_error()

def generate_answers_for_test_data(model_id):
    # Load the test data
    test_df = pd.read_json('test_data.json', lines=True)
    
    generated_data = []
    for index, row in test_df.iterrows():
        # Parse rubric JSON string into a readable format for the prompt
        rubric = json.loads(row['rubric'])  # Converts rubric JSON string to dictionary
        rubric_text = ", ".join(rubric["items"])
        
        prompt = f"Context: {row['task_content']}\n\nQuestion: {row['question']}\n\nRubric: {rubric_text}"
        
        # Generate the exemplar answer
        generated_answer = generate_exemplar_answer(model_id, prompt)
        generated_data.append({
            'question_id': row['question_id'],
            'generated_answer': generated_answer.strip('"'),
            'actual_answer': row['answer'].strip('"')
        })
        print(f"Generated answer for question ID {row['question_id']}")
    
    # Save the generated answers along with actual answers for evaluation
    generated_df = pd.DataFrame(generated_data)
    generated_df.to_json('generated_test_answers.json', orient='records', lines=True)
    print("Generated answers saved to 'generated_test_answers.json'.")

def compute_bertscore(references, candidates):
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()
    
    return {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}

def evaluate_generated_answers():
    # Load the generated answers
    generated_df = pd.read_json('generated_test_answers.json', lines=True)
    references = generated_df['actual_answer'].tolist()
    candidates = generated_df['generated_answer'].tolist()
    
    # Compute BERTScore
    bertscore_scores = compute_bertscore(references, candidates)
    
    # Return only BERTScore
    return {
        'BERTScore': bertscore_scores
    }

if __name__ == "__main__":
    fine_tuned_model_id = 'ft:gpt-4o-mini-2024-07-18:personal::AMkfu4yb' 
    generate_answers_for_test_data(fine_tuned_model_id)
    
    # Evaluate the generated answers using only BERTScore
    results = evaluate_generated_answers()
    print("\nBERTScore Evaluation Results:")
    print(results)

