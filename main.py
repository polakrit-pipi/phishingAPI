# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from classifier import phishing_score
from bilstm_model import BiLSTMModel
from llm_summary import generate_summary

app = FastAPI(title="Phishing URL Analyzer")

# -----------------------------
# Load BiLSTM model + preprocessing objects
# -----------------------------
bilstm = BiLSTMModel(
    model_path="models/model.keras",
    tokenizer_path="models/tokenizer-2.joblib",
    scaler_path="models/scaler-2.joblib",
    labelencoder_path="models/labelencoder-2.joblib"
)

# -----------------------------
# Request model
# -----------------------------
class URLRequest(BaseModel):
    url: str
    call_llm: Optional[bool] = False

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return {"message": "Phishing Analyzer API with LLM summary is online."}

# -----------------------------
# Analyze endpoint
# -----------------------------
@app.post("/analyze")
def analyze(request: URLRequest):
    url = request.url

    # --- Heuristic / rule-based score ---
    try:
        score, reasons, features, host, scheme = phishing_score(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during feature extraction: {e}")

    # --- BiLSTM prediction ---
    try:
        bilstm_label, bilstm_prob = bilstm.predict(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in BiLSTM prediction: {e}")

    response: Dict[str, Any] = {
        "url": url,
        "scheme": scheme,
        "host": host,
        "heuristic_score": score,
        "reasons": reasons,
        "features": features,
        "bilstm": {
            "label": bilstm_label,
            "probability": float(bilstm_prob)
        }
    }

    # --- Optional LLM summary ---
    if request.call_llm:
        try:
            summary_text = generate_summary(
                url, score, reasons, bilstm_label, bilstm_prob,
                features, host, scheme
            )
            response["llm_result"] = summary_text
        except Exception as e:
            response["llm_summary_error"] = str(e)

    return response
