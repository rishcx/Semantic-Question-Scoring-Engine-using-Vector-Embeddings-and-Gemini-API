import requests
import time

def evaluate_answer_with_llm(question, student_answer):
    prompt = f"""Evaluate this student's answer to the question and provide a score out of 5.
Consider the following criteria:
1. Accuracy of information (2 points)
2. Completeness of answer (1 point)
3. Clarity and organization (1 point)
4. Use of appropriate terminology (1 point)

Question: {question}
Student's Answer: {student_answer}

Provide your evaluation in this format:
Score: X/5
Feedback: [Your detailed feedback here]"""

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # Increased timeout for larger model
            )
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                return result['response']
            else:
                print(f"Attempt {attempt + 1}: Unexpected API response format:", result)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return "Score: 0/5\nFeedback: Error in evaluation - Invalid response format"
                
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error calling Ollama API: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "Score: 0/5\nFeedback: Could not connect to evaluation service"
            
        except (KeyError, ValueError) as e:
            print(f"Attempt {attempt + 1}: Error parsing Ollama response: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "Score: 0/5\nFeedback: Error in processing evaluation"

    return "Score: 0/5\nFeedback: Failed to get evaluation after multiple attempts"

def extract_score(evaluation):
    try:
        # Split into lines and find the score line
        lines = evaluation.split('\n')
        score_line = next((line for line in lines if line.startswith('Score:')), None)
        
        if not score_line:
            print("No score line found in evaluation")
            return 0
            
        # Extract the score
        score_text = score_line.split('/')[0].split(':')[1].strip()
        score = float(score_text)
        
        # Validate score is between 0 and 5
        if 0 <= score <= 5:
            return score
        else:
            print(f"Invalid score value: {score}")
            return 0
            
    except Exception as e:
        print(f"Error extracting score: {e}")
        return 0 