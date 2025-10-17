import os
import json
import textwrap
from typing import Dict, Any

def generate_summary(url: str, score: float, reasons: list, label: str, prob: float, features: Dict[str, Any], host: str, scheme: str):
    """
    Generate an LLM-based phishing summary in JSON format.
    Returns dict with keys: verdict, reason_list, summary.
    """
    key = os.environ.get("OPENAI_API_KEY")

    # -------------------
    # 🧩 Fallback (no LLM)
    # -------------------
    if not key:
        verdict = "Likely Phishing" if label.lower() == "phishing" or score > 0.6 else "Likely Safe"
        summary = (
            f"URL: {url} ({verdict}). ตรวจพบสัญญาณความเสี่ยง {len(reasons)} รายการ "
            f"เช่น {', '.join(reasons[:3])} ... "
            f"โมเดล BiLSTM ประเมินความมั่นใจ {prob*100:.1f}%."
        )
        return {
            "verdict": verdict,
            "reason_list": reasons,
            "summary": summary
        }

    # -------------------
    # 🧠 LLM Mode
    # -------------------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        # Build the same prompt format as your sample
        prompt = f"""
Analyze this URL for phishing potential:

URL: {url}
Host: {host}
Scheme: {scheme}

AI Prediction: {label} (confidence={prob:.2f})
Rule-based Risk Score: {score}

Triggered Alerts:
- {"\n- ".join(reasons)}

Technical Features:
- Digit count: {features.get('digit_count')}
- URL length: {features.get('url_length')}
- URL entropy: {features.get('url_entropy', 0):.2f}
- External links: {len(features.get('hrefs', []))}
- Images: {len(features.get('imgs', []))}
- Scripts: {len(features.get('scripts', []))}
- Forms: {len(features.get('forms', []))}

Provide a concise analysis (2-3 sentences) and final verdict as 'Likely Phishing' or 'Likely Safe'.
Return JSON format:
{{
    "verdict": "...",
    "reason_list": ["...","..."],
    "summary": "..."
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_text = response.choices[0].message.content.strip()

        # Clean triple-backticks
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        parsed = json.loads(raw_text)
        return parsed

    except Exception as e:
        return {
            "verdict": "Analysis Error",
            "reason_list": [f"LLM processing failed: {str(e)}"],
            "summary": "Could not complete AI analysis"
        }
