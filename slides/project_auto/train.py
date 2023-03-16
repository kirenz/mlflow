import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Enable autologging
mlflow.sklearn.autolog()

# Argument parsing
parser = argparse.ArgumentParser(description="Train a model with autologging")
parser.add_argument("--experiment_name", type=str, default="My MLflow Experiment",
                    help="Name of the experiment to log the run in")
args = parser.parse_args()

# Load and split the dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Train the model and log the run within the experiment
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Log the accuracy metric (optional, since autologging will capture it automatically)
    mlflow.log_metric("accuracy", accuracy)

    # Add tags to the run
    tags = {
        "model_type": "RandomForest",
        "dataset": "Iris",
        "purpose": "Example"
    }
    mlflow.set_tags(tags)


# How to run the script in bash
# mlflow run . --experiment-name "My Custom Experiment Name"
