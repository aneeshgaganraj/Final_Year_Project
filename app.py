from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, predict_claim
from evaluation import evaluate_metrics

app = Flask(__name__)
CORS(app)

# Load your financial misinformation detection model
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    claim = data.get('claim')
    
    if not claim:
        return jsonify({"error": "No claim provided"}), 400
    
    prediction, explanation = predict_claim(model, claim)
    
    return jsonify({
        "prediction": prediction,
        "explanation": explanation
    })

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    predictions = data.get('predictions')
    references = data.get('references')

    if not predictions or not references:
        return jsonify({"error": "Missing predictions or references"}), 400

    results = evaluate_metrics(predictions, references)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)