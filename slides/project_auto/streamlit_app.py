import os
import mlflow
import mlflow.pyfunc
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


# Calculate the minimum and maximum values for each feature with a 30% range expansion
min_values = iris_df.min() - (iris_df.min() * 0.3)
max_values = iris_df.max() + (iris_df.max() * 0.3)


# Load the registered model from the MLflow Model Registry
MODEL_NAME = "IrisClassifier"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Define the class names for the Iris dataset
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Add a function that returns the image path based on the class name:


def get_image_path(class_name):
    # Define the directory containing the images
    image_dir = "images"
    # Return the image path for the given class name
    return os.path.join(image_dir, f"{class_name}.jpg")


# Streamlit app title
st.title("Iris Classifier")

# Input fields for the features using sliders
sepal_length = st.slider("Sepal Length", min_value=float(
    min_values[0]), max_value=float(max_values[0]), value=1.0)
sepal_width = st.slider("Sepal Width", min_value=float(
    min_values[1]), max_value=float(max_values[1]), value=1.0)
petal_length = st.slider("Petal Length", min_value=float(
    min_values[2]), max_value=float(max_values[2]), value=1.0)
petal_width = st.slider("Petal Width", min_value=float(
    min_values[3]), max_value=float(max_values[3]), value=1.0)

# Button to make a prediction
if st.button("Predict"):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = CLASS_NAMES[int(prediction[0])]

    # Display the prediction results
    st.write(f"Prediction: {predicted_class} (class {int(prediction[0])})")

    # Display the image for the predicted class
    image_path = get_image_path(predicted_class)
    st.image(
        image_path, caption=f"{predicted_class.capitalize()} image", use_column_width=True)
