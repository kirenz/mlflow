import argparse
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import mlflow
import mlflow.sklearn


def load_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data


def train(data, model_type):
    X = data.drop(columns=['target'])
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(
            f"Invalid model type '{model_type}'. Supported model type: 'RandomForest'")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return model, y_test, y_pred, y_pred_proba


def evaluate(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    return accuracy, logloss


def main(data_path, model_type):
    # Create or set the experiment
    experiment_name = "My new MLflow Experiment"
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass  # Experiment already exists, continue
    mlflow.set_experiment(experiment_name)
    data = load_data()
    if data_path:
        data.to_csv(data_path, index=False)

    with mlflow.start_run():
        model, y_test, y_pred, y_pred_proba = train(data, model_type)
        accuracy, logloss = evaluate(y_test, y_pred, y_pred_proba)

        print(f"Accuracy: {accuracy}")
        print(f"Log Loss: {logloss}")

        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("log_loss", logloss)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a model on the Iris dataset.')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to save the Iris dataset as a CSV file (optional).')
    parser.add_argument('--model', type=str, default='RandomForest',
                        help="Type of model to train. Supported model type: 'RandomForest'")
    args = parser.parse_args()

    main(args.data, args.model)
