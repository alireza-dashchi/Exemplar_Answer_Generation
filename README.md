# Exemplar Answer Generation with OpenAI API

This project generates exemplar answers for educational tasks using OpenAI's fine-tuned language model. The generated answers help educators by providing high-quality references for evaluating student responses. The project integrates the OpenAI API to fine-tune a language model with educational tasks, questions, and rubrics.

## Table of Contents
- [Project Overview](#project-overview)
  - [Core Objectives](#core-objectives)
  - [Solution Structure](#solution-structure)
- [Technical Requirements](#technical-requirements)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Virtual Environment](#set-up-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Options in `main.py`](#options-in-mainpy)
  - [Step-by-Step Workflow](#step-by-step-workflow)
- [Evaluation](#evaluation)
  - [Evaluation Logic](#evaluation-logic)
  - [Evaluation Result](#evaluation-result)
  - [BERTScore](#bertscore)
  - [Evaluation Process](#evaluation-process)
  - [Analysis and Thought Process](#analysis-and-thought-process)
  - [Considerations](#considerations)
- [Future Improvements](#future-improvements)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Test Coverage](#test-coverage)
  - [Edge Cases and Error Handling](#edge-cases-and-error-handling)
- [Contact](#contact)


---

## Project Overview

Cura Education aims to assist teachers by providing automatically generated exemplar answers for educational tasks. These answers serve as benchmarks to improve the efficiency and consistency of grading. This project uses a fine-tuned version of OpenAI's GPT-4o model, specifically trained on provided educational data, to generate exemplar answers that align with task rubrics and criteria.

### Core Objectives
1. Integrate with the OpenAI API to fine-tune a language model for generating exemplar answers.
2. Evaluate generated answers against rubrics and sample training data.
3. Develop an evaluation system to assess the quality of generated answers.

---

### Solution Structure

The solution consists of several key components:

1. **Data Preparation**:
   - Splits the provided dataset into training, testing, and holdout sets using `scikit-learn`.
   - Formats the training data into the required format for fine-tuning the OpenAI model.
   - Data includes task content, questions, rubrics, and exemplar answers.

2. **Model Fine-Tuning**:
   - Uploads the formatted training data to OpenAI's servers using the OpenAI API.
   - Initiates the fine-tuning process for the GPT-4o-mini model with the uploaded data.
   - Monitors the fine-tuning job and retrieves the fine-tuned model ID upon completion.

3. **Exemplar Answer Generation**:
   - Uses the fine-tuned model to generate exemplar answers based on new inputs.
   - Inputs include task context, question, and assessment rubric.

4. **Evaluation**:
   - Generates exemplar answers for the test data using the fine-tuned model.
   - Evaluates the generated answers against actual exemplar answers using BERTScore.
   - Fine-tuned the model with test data after initial training.
   - Optimized parameters (e.g., system instructions) for improved performance.
   - Combined training and test data for a final round of fine-tuning.
   - Tested the model on unseen data to assess generalization capabilities.

5. **Testing**:
   - Includes automated tests using `pytest` to ensure code correctness and handle edge cases.
   - Tests cover data preparation, OpenAI API integration, and the main process flow.
---

## Technical Requirements

1. **Python version**: Tested on Python 3.9.
2. **Core Libraries**:
   - `transformers`, `openai`, `pandas`, `scikit-learn`, `bert-score`
3. **OpenAI API Key**: Required to fine-tune the model and generate responses.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alireza-dashchi/exemplar-answer-generation
   cd exemplar-answer-generation

### Set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # For macOS/Linux
.\venv\Scripts\activate    # For Windows


### Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root and add your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

### Project Structure

```plaintext
.
├── main.py                # Main entry point for running the application
├── data_preparation.py    # Handles data splitting and formatting
├── openai_integration.py  # Integrates OpenAI API for fine-tuning and generating answers
├── evaluation.py          # Evaluates the quality of generated answers using BERTScore
├── test                   # Directory for test files
│   ├── test_main.py       # Tests for main.py
│   ├── test_data_preparation.py # Tests for data_preparation.py
│   ├── test_openai_integration.py # Tests for OpenAI API integration
├── requirements.txt       # Dependencies for the project
└── README.md              # Project documentation
```

### Usage
Run the main program:

```bash
python3 main.py
```

### Options in `main.py`:
- **Generate Exemplar Answer**: Generate an exemplar answer for a question by providing task context, question, and rubric.
- **Tune a New Model**: Fine-tune a new model on the provided training data.
- **Evaluate Model**: Generate and evaluate exemplar answers using BERTScore.

### Step-by-Step Workflow:
1. **Prepare Data**: `data_preparation.py` will split and format data into training and testing sets.
2. **Fine-Tune Model**: Upload formatted data for fine-tuning using OpenAI API.
3. **Generate Exemplar Answers**: Generate answers based on the fine-tuned model.
4. **Evaluate**: Assess model-generated answers with BERTScore.

### Evaluation

**Evaluation Logic**  
The evaluation focuses on assessing how closely the model-generated exemplar answers match the actual exemplar answers in the test set. The primary evaluation metric used is BERTScore, which measures the similarity of two texts at the semantic level.

#### Evaluation Results 
**Evaluation Results on Test Data**
```bash
    "BERTScore": 
        "precision": 0.88999342918396,
        "recall": 0.8835974335670471,
        "f1": 0.8866317868232727
```

**Evaluation Results on Unseen Data**
```bash
  "BERTScore": 
    "precision": 0.8925970196723938,
    "recall": 0.8868222236633301,
    "f1": 0.8895487785339355
```

#### BERTScore

**What is BERTScore?**  
- BERTScore leverages pre-trained language models like BERT to compute similarity scores between sentences.
- It provides precision, recall, and F1 scores based on token embeddings.

**Why BERTScore?**  
- It captures semantic similarities better than traditional metrics like BLEU or ROUGE.
- It’s suitable for evaluating open-ended text generation tasks.

#### Evaluation Process

**Generate Answers for Test Data**  
- The fine-tuned model generates exemplar answers for each question in the test set.
- The prompts include the task content, question, and a parsed version of the rubric.

**Compute BERTScore**  
- The generated answers (candidates) are compared against the actual exemplar answers (references).
- BERTScore computes precision, recall, and F1 scores for each pair.

**Aggregate Results**  
- The average precision, recall, and F1 scores are calculated across all test samples.
- These metrics provide an overall assessment of the model's performance.

#### Analysis and Thought Process

**Interpreting BERTScore Metrics**  
- **Precision**: How much of the generated answer is relevant to the reference.
- **Recall**: How much of the reference is covered by the generated answer.
- **F1 Score**: The harmonic mean of precision and recall.

**Quality Assessment**  
- Higher BERTScore F1 scores indicate that the generated answers are semantically similar to the actual answers.
- By analyzing individual scores, we can identify questions where the model performs well or needs improvement.

**Out-of-Sample Evaluation**  
- The holdout set can be used to evaluate the model's performance on unseen data.
- This helps assess the model's generalization capability and readiness for real-world deployment.

#### Considerations

- Since the data set is relatively small, results may vary, and overfitting is possible.
- Additional evaluation methods (e.g., human evaluation, other metrics) can complement BERTScore.

### Future Improvements
Due to limited time, I did not implement some features that could enhance the model's performance. In the future, we can utilize the built-in features of the OpenAI API named **Evaluations**, which offer several evaluation criteria:

- **Factuality**: Check if the content is factually accurate.
- **Semantic Similarity**: Compare generated text to the reference.
- **Custom Prompt**: Create a test criterion by writing your own custom prompt.
- **Sentiment**: Identify the emotional tone of the model's response.
- **String Check**: Verify if the model's response includes specific strings.
- **Valid JSON or XML**: Ensure the model's response is valid JSON or XML.
- **Matches Schema**: Ensure the model's response follows the specified structure.
- **Criteria Match**: Assess if the model's response matches predefined criteria.
- **Text Quality**: Assess response quality with BLEU, ROUGE, or Cosine similarity algorithms.

Additionally, we can:

- **Increase Training Data**: Expanding the dataset can help the model learn a wider range of patterns and improve performance.
- **Incorporate Human Expert Evaluation**: Involving educators to evaluate and provide feedback on generated answers can enhance the model's quality.

### Testing

Automated tests are included to ensure that each component of the project works correctly and handles potential edge cases.

#### Running Tests
Ensure that you have `pytest` installed:

pip install pytest

Run all tests from the project root directory:

pytest test/

#### Test Coverage

**test_main.py**  
- Tests the main process flow, including data preparation, fine-tuning initiation, and model retrieval.
- Uses mocking to simulate OpenAI API responses.

**test_data_preparation.py**  
- Tests data splitting and formatting functions.
- Includes tests for empty data, invalid JSON formats, and formatting of training examples.

**test_openai_integration.py**  
- Tests OpenAI API integration functions.
- Handles cases like non-existent files, invalid training file IDs, and API errors.

#### Edge Cases and Error Handling
The tests cover scenarios such as:
- Empty input data files.
- Invalid JSON formats in data files.
- API failures or incorrect API usage.
- Ensuring that the code gracefully handles exceptions and provides meaningful error messages.


### Contact
For questions, please contact [alirezadashchi@gmail.com](mailto:alirezadashchi@gmail.com).
