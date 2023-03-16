"""
Local deployment

"""

import mlflow


# Replace these placeholders with the appropriate values for your use case
MODEL_NAME = "IrisClassifier"
MODEL_VERSION = 3  # Replace with the desired version number
RUN_ID = "da22e09d42d94b0db3e7981e919d3b1d"
MODEL_URI = f"runs:/{RUN_ID}/model"

# Optional: Set the tracking URI to the local 'mlruns' folder (if not already set)
# mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# Register the model
model_details = mlflow.register_model(MODEL_URI, MODEL_NAME)

# Get the model version
model_version = model_details.version

# Update the model version stage to "Staging"
mlflow.tracking.MlflowClient().transition_model_version_stage(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    stage="Staging"
)
