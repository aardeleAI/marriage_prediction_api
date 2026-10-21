from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Model loading

model = joblib.load("models/reduced_model_10.pkl")

selected_features = ["enjoy_travel", "harmony", "dreams", "happy", "marriage", "anxieties",
                     "Aggro_argue", "hate_subjects", "idk_what's_going_on", "good_to_leave_home"]

class MarriageFeatures(BaseModel):
    enjoy_travel: int
    harmony: int
    dreams: int
    happy: int
    marriage: int
    anxieties: int
    Aggro_argue:int
    hate_subjects: int
    idk_what_s_going_on: int
    good_to_leave_home: int

@app.get("/")
def root():
    return {"message": "Marriage prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: MarriageFeatures):

    df =pd.DataFrame([[features.enjoy_travel,
                       features.harmony,
                       features.dreams,
                       features.happy,
                       features.marriage,
                       features.anxieties,
                       features.Aggro_argue,
                       features.hate_subjects,
                       features.idk_what_s_going_on,
                       features.good_to_leave_home]],
                     columns=["enjoy_travel", "harmony", "dreams", "happy",
                              "marriage", "anxieties", "Aggro_argue",
                              "hate_subjects", "idk_what's_going_on", "good_to_leave_home"])

    # Model probabilities

    probability = model.predict_proba(df)[0]

    stable_prob = float(probability[0])
    divorce_prob = float(probability[1])

    # Prediction

    prediction = 1 if stable_prob >= 0.5 else 0

    # Interpretation message

    if prediction == 1:
        message = "Your marriage has a good chance of lasting!"
    else:
        message = "Your relationship has some vulnerabilities"

    return {"prediction": int(prediction), "stable_prob": stable_prob,
            "divorce_prob": divorce_prob, "message": message}
