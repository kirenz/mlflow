import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np


# Define the class names for the Iris dataset
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


# Load the registered model from the MLflow Model Registry
MODEL_NAME = "IrisClassifier"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Define a FastAPI app
app = FastAPI()

# Create a Pydantic schema for input data


class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define an endpoint for making predictions


@app.post("/predict")
def predict(data: InputData):
    # Convert input data to a NumPy array
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]])

    # Make a prediction using the MLflow model
    try:
        prediction = model.predict(input_data)
        predicted_class = CLASS_NAMES[int(prediction[0])]
        return {"prediction": int(prediction[0]), "class_name": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
