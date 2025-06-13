from flask import Flask, render_template, request, jsonify
from evaluator import evaluate_answer_with_llm, extract_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    logger.info("Received evaluation request")
    data = request.get_json()
    
    if not data or 'qa_list' not in data:
        logger.error("No questions and answers provided")
        return jsonify({'error': 'No questions and answers provided'}), 400
    
    qa_list = data['qa_list']
    logger.info(f"Processing {len(qa_list)} Q&A pairs")
    
    try:
        # Evaluate each answer
        results = []
        total_score = 0
        max_score = len(qa_list) * 5

        for idx, qa in enumerate(qa_list):
            logger.info(f"Evaluating Q&A pair {idx + 1}")
            if 'question' not in qa or 'answer' not in qa:
                logger.error(f"Missing question or answer in pair {idx + 1}")
                return jsonify({'error': 'Each Q&A pair must have both question and answer'}), 400
                
            # Get LLM evaluation
            logger.info(f"Calling LLM for pair {idx + 1}")
            evaluation = evaluate_answer_with_llm(qa['question'], qa['answer'])
            logger.info(f"Got evaluation for pair {idx + 1}: {evaluation}")
            
            score = extract_score(evaluation)
            logger.info(f"Extracted score for pair {idx + 1}: {score}")
            total_score += score
            
            results.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'evaluation': evaluation,
                'score': score
            })
        
        # Calculate percentage
        percentage = (total_score/max_score) * 100
        logger.info(f"Final scores - Total: {total_score}, Max: {max_score}, Percentage: {percentage}%")
        
        response_data = {
            'results': results,
            'total_score': total_score,
            'max_score': max_score,
            'percentage': round(percentage, 2)
        }
        logger.info("Sending response back to client")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 