"""
Local deployment

"""

import numpy as np
import mlflow.pyfunc

# Load the registered model
MODEL_NAME = "IrisClassifier"
MODEL_VERSION = 3  # Replace with the desired version number

loaded_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")

# Use the model for prediction

# Example input for an Iris flower
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(sample_input)
print("Prediction:", prediction)
