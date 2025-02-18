# Third party imports
from flask import Flask, request, jsonify
import pandas as pd

# Local imports
from src.score import ModelScoring

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.get_json()
    
    # Assuming the data comes in as a list of rows (each row is a list of feature values)
    # Convert the list of rows into a Pandas DataFrame
    df = pd.DataFrame(data['features'])
    
    # Make the prediction using the loaded model
    predictions = ModelScoring.execute(df)
    
    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Expose the API on port 5000
