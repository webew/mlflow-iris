import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/iris-classifier@champion")

app = FastAPI()


CLASSES = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def accueil():
    return {"message": "API Iris en ligne"}

@app.post("/predict")
def predict(features: IrisFeatures):
    x = [[features.sepal_length, features.sepal_width,
          features.petal_length, features.petal_width]]
    prediction = int(model.predict(x)[0])
    return {"classe": prediction, "espece": CLASSES[prediction]}

from mlflow import MlflowClient

client = MlflowClient()
champion = client.get_model_version_by_alias("iris-classifier", "champion")


@app.get("/info")
def info():
    return {"modele": champion.name, "version": champion.version}