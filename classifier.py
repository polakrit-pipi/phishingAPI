import tldextract, re, requests
from bs4 import BeautifulSoup

def phishing_score(url):
    features = {}
    ext = tldextract.extract(url)
    host = f"{ext.domain}.{ext.suffix}"
    scheme = "https" if url.startswith("https") else "http"

    features["https"] = url.startswith("https")
    features["ip_in_host"] = bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))
    features["has_at"] = "@" in url
    features["long_url"] = len(url) > 75
    features["suspicious_subdomain"] = len(ext.subdomain.split(".")) > 2

    reasons = []
    try:
        html = requests.get(url, timeout=3).text
        soup = BeautifulSoup(html, "html.parser")
        if soup.find_all("form", action=re.compile(r"http")):
            features["external_form_action"] = True
        else:
            features["external_form_action"] = False
    except Exception:
        features["external_form_action"] = None

    score = sum(int(v) for v in features.values() if isinstance(v, bool)) / len(features)
    reasons = [k for k, v in features.items() if v]
    return score, reasons, features, host, scheme
