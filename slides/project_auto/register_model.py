import mlflow

# Set the tracking URI if you are not using the default one
# mlflow.set_tracking_uri("http://your_tracking_uri:port")

# Set the experiment you used for the run
mlflow.set_experiment("My Custom Experiment Name")

# Get the run_id for the run you want to register the model from
# Replace the id with your actual run ID of the successful run
your_run_id = "da22e09d42d94b0db3e7981e919d3b1d"

# Register the model
model_uri = f"runs:/{your_run_id}/model"
model_name = "IrisClassifier"
mlflow.register_model(model_uri, model_name)
