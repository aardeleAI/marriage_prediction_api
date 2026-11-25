from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Model loading

model = joblib.load("models/reduced_model_6.pkl")

selected_features = ["begin_correct", "trust", "anxieties",
                     "idk_what's_going_on", "good_to_leave_home", "accusations"]

class MarriageFeatures(BaseModel):
    begin_correct: int
    trust: int
    anxieties: int
    idk_what_s_going_on: int
    good_to_leave_home: int
    accusations: int

@app.get("/")
def root():
    return {"message": "Marriage prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MarriageFeatures):

    df =pd.DataFrame([[features.begin_correct,
                       features.trust,
                       features.anxieties,
                       features.idk_what_s_going_on,
                       features.good_to_leave_home,
                       features.accusations]],
                     columns=["begin_correct", "trust", "anxieties",
                     "idk_what's_going_on", "good_to_leave_home", "accusations"])

    # Model probabilities

    probability = model.predict_proba(df)[0]

    stable_prob = float(probability[0])
    divorce_prob = float(probability[1])

    # Prediction

    prediction = 1 if divorce_prob >= 0.5 else 0

    # Interpretation message

    if prediction == 1:
        message = "Your relationship has some vulnerabilities"
    else:
        message = "Your marriage has a good chance of lasting!"

    return {"prediction": int(prediction), "stable_prob": stable_prob,
            "divorce_prob": divorce_prob, "message": message}
