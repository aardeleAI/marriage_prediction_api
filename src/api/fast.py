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

    df =pd.DataFrame([features.dict()])

    # Missing features check
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        return {"error": f"Missing features: {missing_features}"}

    # Class prediction (class 0 for stable marriage and class 1 for risky marriage)
    prediction = model.predict(df)[0]

    # Interpretation message
    if prediction == 0:
        message = "Your marriage has a good chance of lasting!"
    else:
        message = "Your relationship has some vulnerabilities"

    return {"prediction": int(prediction), "message": message}
