# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from feature_utils import fetch_html, phishing_score
from predict_utils import predict_url
import json
from model_utils import model, client  # client สำหรับ LLM ถ้ามี
import numpy as np

# --- เพิ่มสำหรับ PhishTank ---
import gzip, csv, io, requests

def load_phishtank_database():
    print("[INFO] Downloading PhishTank dataset...")
    url = "http://data.phishtank.com/data/online-valid.csv.gz"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = gzip.decompress(r.content)
    csv_data = csv.DictReader(io.StringIO(data.decode()))
    urls = {row['url'] for row in csv_data}
    print(f"[INFO] Loaded {len(urls)} phishing URLs from PhishTank.")
    return urls

# โหลดครั้งเดียวตอนเริ่มแอป
phishtank_cache = load_phishtank_database()


app = FastAPI()

class URLRequest(BaseModel):
    url: str
    call_llm: bool = True

@app.post("/analyze")
def analyze(request: URLRequest):
    url = request.url
    html = fetch_html(url)
    score, reasons, features, host, scheme = phishing_score(url, html)

    # --- เช็ค PhishTank ---
    if url in phishtank_cache:
        reasons.append("URL found in PhishTank database")

    # BiLSTM prediction
    bilstm_label, bilstm_prob_array = predict_url(url)
    label_idx = int(np.argmax(bilstm_prob_array))
    bilstm_prob = float(bilstm_prob_array[label_idx])

    # LLM analysis
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
- Digit count: {features.get('digit_count')}
- URL length: {features.get('url_length')}
- URL entropy: {features.get('url_entropy'):.2f}
- External links: {len(features['hrefs'])}
- Images: {len(features['imgs'])}
- Scripts: {len(features['scripts'])}
- Forms: {len(features['forms'])}

Provide a concise analysis (2-3 sentences) and final verdict as 'Likely Phishing' or 'Likely Safe'.
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
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"): raw_text = raw_text[3:-3].strip()
            llm_result = json.loads(raw_text)
        except Exception as e:
            llm_result = {"verdict":"Analysis Error","reason_list":[str(e)],"summary":"Could not complete AI analysis"}

    return {"url": url, "score": score, "reasons": reasons, "features": features,
            "bilstm_label": bilstm_label, "bilstm_prob": bilstm_prob,
            "llm_result": llm_result, "host": host, "scheme": scheme}

@app.get("/")
def root():
    return {"message":"Phishing URL Analyzer API is running!", "status":"active"}
