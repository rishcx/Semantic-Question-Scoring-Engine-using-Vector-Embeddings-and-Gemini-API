from PyPDF2 import PdfReader
import re
import requests
import chromadb
from chromadb.config import Settings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

def extract_qa_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Split text into lines
    lines = text.split('\n')
    qa_pairs = []
    current_question = None
    current_answer = []
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if line starts with a number followed by a dot or parenthesis
        if re.match(r'^\d+[\.\)]', line):
            # If we have a previous question and answer, save them
            if current_question:
                qa_pairs.append({
                    "question": current_question,
                    "answer": " ".join(current_answer).strip()
                })
            # Start new question
            current_question = line
            current_answer = []
        elif current_question:  # If we have a question, this line is part of the answer
            current_answer.append(line)
    
    # Add the last question-answer pair
    if current_question:
        qa_pairs.append({
            "question": current_question,
            "answer": " ".join(current_answer).strip()
        })
    
    return qa_pairs

def get_embedding(text):
    # Preprocess the text before getting embeddings
    processed_text = preprocess_text(text)
    # Replace with your Ollama model endpoint and model name
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "llama3", "prompt": processed_text}
    )
    return response.json()["embedding"]

chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))
collection = chroma_client.get_or_create_collection("answers")

def store_reference_answers(qa_list):
    try:
        for idx, qa in enumerate(qa_list):
            emb = get_embedding(qa["answer"])
            collection.add(
                embeddings=[emb],
                documents=[qa["answer"]],
                metadatas=[{"question": qa["question"], "id": idx}],
                ids=[str(idx)]
            )
        print("\nSuccessfully stored all answers in vector database!")
    except Exception as e:
        print(f"\nError storing in vector database: {e}")

def evaluate_answer_with_llm(question, student_answer):
    prompt = f"""Evaluate this student's answer to the question and provide a score out of 5.
Consider the following criteria:
1. Accuracy of information (2 points)
2. Completeness of answer (1 point)
3. Clarity and organization (1 point)
4. Use of appropriate terminology (1 point)

Question: {question}
Student's Answer: {student_answer}

Provide only the score in this format:
Score: X/5"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        if 'response' in result:
            return result['response']
        else:
            print("Unexpected API response format:", result)
            return "Score: 0/5"
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "Score: 0/5"
    except (KeyError, ValueError) as e:
        print(f"Error parsing Ollama response: {e}")
        return "Score: 0/5"

if __name__ == "__main__":
    # Step 1: Extract Q&A from PDF
    qa_list = extract_qa_from_pdf("quesans.pdf")
    print("\nEvaluating Student Answers from PDF:")
    print("----------------------------------------")
    
    # Evaluate each answer
    student_results = []
    total_score = 0
    max_score = len(qa_list) * 5  # 5 points per question

    for idx, qa in enumerate(qa_list):
        print(f"\nQuestion {idx + 1}:")
        print(f"Q: {qa['question']}")
        print(f"Student's Answer: {qa['answer']}")
        
        # Get LLM evaluation
        evaluation = evaluate_answer_with_llm(qa['question'], qa['answer'])
        
        # Store results
        result = {
            "question": qa['question'],
            "student_answer": qa['answer'],
            "evaluation": evaluation
        }
        student_results.append(result)
        
        # Extract score from evaluation (assuming format "Score: X/5")
        try:
            score_line = [line for line in evaluation.split('\n') if line.startswith('Score:')][0]
            score = float(score_line.split('/')[0].split(':')[1].strip())
            total_score += score
        except:
            print("Warning: Could not parse score from evaluation")
            score = 0

    # Print final results
    print("\n=== FINAL RESULTS ===")
    print(f"Total Score: {total_score}/{max_score}")
    print(f"Percentage: {(total_score/max_score)*100:.2f}%")
    
    print("\nScores for each question:")
    print("-----------------")
    for idx, result in enumerate(student_results):
        print(f"Question {idx + 1}: {result['evaluation']}")
