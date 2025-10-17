from fastapi import FastAPI
from pydantic import BaseModel
from feature_utils import fetch_html, phishing_score
from predict_utils import predict_url
from model_utils import model, client  # client สำหรับ LLM ถ้ามี
import numpy as np
import json
import gzip, csv, io, requests

# -----------------------------
# Helper: แปลง numpy type เป็น native Python
# -----------------------------
def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

# -----------------------------
# โหลดฐานข้อมูล PhishTank
# -----------------------------
def load_phishtank_database():
    print("[INFO] Downloading PhishTank dataset...")
    url = "http://data.phishtank.com/data/online-valid.csv.gz"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = gzip.decompress(r.content)
        csv_data = csv.DictReader(io.StringIO(data.decode()))
        urls = {row["url"] for row in csv_data}
        print(f"[INFO] Loaded {len(urls)} phishing URLs from PhishTank.")
        return urls
    except Exception as e:
        print(f"[WARN] Failed to load PhishTank: {e}")
        return set()

phishtank_cache = load_phishtank_database()

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

class URLRequest(BaseModel):
    url: str
    call_llm: bool = True

# -----------------------------
# วิเคราะห์ URL หลัก
# -----------------------------
@app.post("/analyze")
def analyze(request: URLRequest):
    url = request.url.strip()
    html = fetch_html(url)
    score, reasons, features, host, scheme = phishing_score(url, html)

    # --- เช็ค PhishTank ---
    if url in phishtank_cache:
        reasons.append("URL found in PhishTank database")

    # --- BiLSTM prediction ---
    bilstm_label, bilstm_prob_array = predict_url(url)
    label_idx = int(np.argmax(bilstm_prob_array))
    bilstm_prob = float(bilstm_prob_array[label_idx])

    # --- LLM analysis ---
    llm_result = None
    if request.call_llm and client:
        prompt = f"""
Analyze this URL for phishing potential:

URL: {url}
Host: {host}
Scheme: {scheme}

AI Prediction: {bilstm_label} (confidence={bilstm_prob:.2f})
Rule-based Risk Score: {score}/15

Triggered Alerts:
- {"\n- ".join(reasons)}

Technical Features:
- Digit count: {features.get('digit_count', 0)}
- URL length: {features.get('url_length', 0)}
- URL entropy: {features.get('url_entropy', 0):.2f}
- External links: {len(features.get('hrefs', []))}
- Images: {len(features.get('imgs', []))}
- Scripts: {len(features.get('scripts', []))}
- Forms: {len(features.get('forms', []))}

Provide a concise analysis (2–3 sentences) and final verdict as 'Likely Phishing' or 'Likely Safe'.
Return JSON format:
{{
    "verdict": "...",
    "reason_list": ["...","..."],
    "summary": "..."
}}
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
            llm_result = json.loads(raw_text)
        except Exception as e:
            llm_result = {
                "verdict": "Analysis Error",
                "reason_list": [str(e)],
                "summary": "Could not complete AI analysis"
            }

    # --- แปลงค่าเป็น native type ก่อน return ---
    result = {
        "url": url,
        "score": int(score),
        "reasons": reasons,
        "features": to_native(features),
        "bilstm_label": bilstm_label,
        "bilstm_prob": float(bilstm_prob),
        "llm_result": llm_result,
        "host": host,
        "scheme": scheme,
    }

    return result

@app.get("/")
def root():
    return {"message": "Phishing URL Analyzer API is running!", "status": "active"}
