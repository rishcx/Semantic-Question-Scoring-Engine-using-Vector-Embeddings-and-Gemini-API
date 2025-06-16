from PyPDF2 import PdfReader
import re

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