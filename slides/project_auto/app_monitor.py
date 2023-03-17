from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
db_username = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

# Create the database URL
DATABASE_URL = f"mysql+pymysql://{db_username}:{db_password}@{db_host}/{db_name}"

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Model
model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")

# Class names
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    predicted_class = Column(String)


Base.metadata.create_all(bind=engine)


class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    predicted_class: str


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    input_features = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
    ]]
    predicted_class = str(CLASS_NAMES[model.predict(input_features)[0]])

    log_prediction(input_features[0], predicted_class)

    return {"predicted_class": predicted_class}


def log_prediction(input_features, predicted_class):
    with SessionLocal() as session:
        prediction = Prediction(
            sepal_length=input_features[0],
            sepal_width=input_features[1],
            petal_length=input_features[2],
            petal_width=input_features[3],
            predicted_class=predicted_class,
        )
        session.add(prediction)
        session.commit()
    session.close()
