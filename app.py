from flask import Flask, request, jsonify
import pandas as pd
from src.score import ModelScoring

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.get_json()

    # Extract feature names and values from the request
    feature_names = data.get('feature_names')
    feature_values = data.get('features')

    # Check if feature_names and features are provided
    if not feature_names or not feature_values:
        return jsonify({"error": "Missing feature_names or features in the request"}), 400

    # Create a DataFrame using the feature names as column names
    df = pd.DataFrame(feature_values, columns=feature_names)

    # Print the DataFrame to the console (for debugging purposes)
    print(df)

    # Make the prediction using the loaded model
    scorer = ModelScoring(df)
    predictions = scorer.execute()

    print(f"predicted values are: {predictions}")

    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions["predicted_diagnosis"].tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
