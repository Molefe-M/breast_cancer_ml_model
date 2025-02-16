# Optimal hidden layer size
opt_hidden_layer_size = 500

# Optimal feature list
opt_feature = ["concave points_worst", "perimeter_worst", "concave points_mean", \
               "radius_worst", "perimeter_mean", "concavity_worst", "concavity_mean", \
                "area_mean", "radius_mean", "area_worst"]

# Model name
model_name = "mlp_breast_cancer.pkl"

# Target label mapping
target_mapping = {"B": 0, "M": 1}

# Predictions mapping
pred_mapping = {0: "B", 1: "M"}