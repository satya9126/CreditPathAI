from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# --------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="CreditPathAI - Credit Default Prediction API",
    description="Predicts credit default risk using XGBoost",
    version="1.0.0"
)

# --------------------------------------------------
# Enable CORS (Required for React frontend)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load trained model & scaler
# --------------------------------------------------
model = joblib.load("creditpath_xgb.pkl")
scaler = joblib.load("creditpath_scaler.pkl")

# --------------------------------------------------
# Input schema (MATCHES DATASET EXACTLY)
# --------------------------------------------------
class Borrower(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_default(data: Borrower):
    """
    Predict probability of credit default and return risk category + action
    """

    # Convert input JSON → DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Apply same scaling used during training
    scaled_input = scaler.transform(input_df)

    # Predict probability of default (class 1)
    probability = model.predict_proba(scaled_input)[0][1]

    # Risk categorization
    if probability < 0.20:
        risk = "Low Risk"
        action = "Send gentle SMS reminder"
    elif probability < 0.40:
        risk = "Moderate Risk"
        action = "Call customer and confirm repayment date"
    elif probability < 0.60:
        risk = "High Risk"
        action = "Offer restructuring or part-payment plan"
    else:
        risk = "Very High Risk"
        action = "Escalate to field visit or legal notice"

    return {
        "default_probability": round(float(probability), 4),
        "risk_category": risk,
        "recommended_action": action
    }
