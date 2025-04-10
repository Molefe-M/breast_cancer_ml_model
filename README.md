# Breast Cancer Prediction API
This project exposes a simple machine learning model as an API for predicting breast cancer based on diagnostic features. It is trained using the Breast Cancer Wisconsin Diagnostic dataset, commonly used for binary classification tasks in medical ML projects.

# Model Summary
* Model Type: Support Vector Machines (SVM)
* Training Library: Scikit-learn
* Target: Predict if a tumor is Malignant (M) or Benign (B)
* Feature Count: 10 selected features chosen for their relevance

# Quickstart (local)
-To work locally, follow these steps:

* Step one: Clone the repo
-git clone https://github.com/Molefe-M/breast_cancer_ml_model.git

* Step two: Create conda environment
-conda create -n breast_cancer_api python=3.9
-conda activate breast_cancer_api
-pip install -r requirements.txt

* Step three: Run the app
python app.py

# Making Predictions
You can send a POST request to the /predict endpoint using Postman or curl.

* Request format
{
  "feature_names": [
    "concave points_worst", "perimeter_worst", "concave points_mean",
    "radius_worst", "perimeter_mean", "concavity_worst", "concavity_mean",
    "area_mean", "radius_mean", "area_worst"
  ],
  "features": [
    [5.1, 3.5, 1.4, 0.2, 0.5, 0.2, 0.5, 0.9, 1.0, 0.1]
  ]
}

* Response format
{
  "predictions": [
    "M"
  ]
}
