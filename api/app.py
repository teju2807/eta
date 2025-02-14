from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'models/model.pkl')
try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    model = None

@app.route('/train', methods=['POST'])
def train():
    from train import train_model
    metrics = train_model()
    return jsonify({"message": "Model trained successfully!", **metrics})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not trained. Train the model first!"})
    data = request.get_json()
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)