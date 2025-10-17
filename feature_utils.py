# feature_utils.py
import re, math
from collections import Counter
from urllib.parse import urlparse
import requests

BRAND_KEYWORDS = ["paypal","apple","amazon","bank","chase","facebook","meta","google","microsoft",
                  "outlook","office365","instagram","line","kbank","scb","krungsri","kplus"]

COMMON_TLDS = set([
 "com","net","org","info","biz","co","io","ai","app","edu","gov","mil","ru","de","uk","cn","fr","jp","br","in","it","es","au","nl","se","no"
])

# -----------------------------
# URL parsing
# -----------------------------
def parse_host_and_scheme(url: str):
    try:
        p = urlparse(url if '://' in url else 'http://' + url)
        return (p.hostname or "").lower(), (p.scheme or "").lower()
    except:
        return "", ""

def is_ip_host(host: str):
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host or ""))

def count_subdomains(host: str):
    if not host: return 0
    return max(0, len(host.split(".")) - 2)

def has_double_slash_in_path(url: str):
    return "//" in (urlparse(url if '://' in url else 'http://' + url).path or "")

def has_tld_in_path(url: str):
    path = (urlparse(url if '://' in url else 'http://' + url).path or "").lower()
    return any(("."+tld) in path for tld in COMMON_TLDS)

def has_symbols_in_domain(host: str):
    return bool(re.search(r"[^a-z0-9\.-]", host or ""))

def domain_prefix_suffix_like_brand(host: str):
    if not host: return False
    first = host.split(".")[0]
    return any(b in first and "-" in first for b in BRAND_KEYWORDS)

def brand_in_path_or_subdomain(host: str, url: str):
    text = ((host or "") + " " + (urlparse(url).path or "") + " " + (urlparse(url).query or "")).lower()
    return any(b in text for b in BRAND_KEYWORDS)

def digit_count(url: str):
    return sum(c.isdigit() for c in url)

def url_length(url: str):
    return len(url)

def url_entropy(url: str):
    if not url: return 0.0
    counts = Counter(url)
    total = len(url)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

# -----------------------------
# HTML Feature extraction
# -----------------------------
def fetch_html(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, timeout=5, headers=headers)
        return r.text
    except Exception as e:
        print(f"⚠️ Could not fetch HTML: {e}")
        return ""

def extract_html_features(html):
    hrefs = re.findall(r'href=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    forms = re.findall(r'<form[^>]+action=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    imgs = re.findall(r'<img[^>]+src=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    scripts = re.findall(r'<script[^>]+src=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    links_tag = re.findall(r'<link[^>]+href=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    meta_keywords = re.findall(r'<meta[^>]+name=[\"\']keywords[\"\'][^>]+content=[\"\'](.*?)[\"\']', html or '', flags=re.IGNORECASE)
    return {'hrefs': hrefs, 'forms': forms, 'imgs': imgs, 'scripts': scripts, 'links_tag': links_tag, 'meta_keywords': meta_keywords}

# -----------------------------
# Heuristic checks
# -----------------------------
def abnormal_links(hrefs):
    return any(h.strip().lower().startswith(('javascript:','mailto:','data:')) for h in hrefs)

def forms_action_abnormal(forms, host):
    for a in forms:
        if a and host not in a and not a.startswith('/') and not a.startswith('#'):
            return True
    return False

def anchors_point_elsewhere(hrefs, host):
    count = sum(1 for h in hrefs if host and host not in h and h.startswith('http'))
    total = max(1, len(hrefs))
    return (count / total) > 0.5

def meta_keyword_mismatch(meta_keywords, host):
    if not meta_keywords: return False
    for kw in meta_keywords:
        if host and host.split('.')[0] not in kw:
            return True
    return False

# -----------------------------
# Rule-based scoring
# -----------------------------
def phishing_score(url, html):
    host, scheme = parse_host_and_scheme(url)
    features = extract_html_features(html)
    score = 0
    reasons = []

    if is_ip_host(host):
        score += 2; reasons.append("Host is an IP address")
    if count_subdomains(host) > 2:
        score += 1; reasons.append("Too many subdomains")
    if has_symbols_in_domain(host):
        score += 1; reasons.append("Suspicious symbols in domain")
    if domain_prefix_suffix_like_brand(host):
        score += 2; reasons.append("Domain mimics brand with hyphens")
    if brand_in_path_or_subdomain(host,url):
        score +=1; reasons.append("Brand keywords in path or subdomain")
    if has_double_slash_in_path(url):
        score +=1; reasons.append("Double slash in path")
    if has_tld_in_path(url):
        score +=1; reasons.append("TLD in path")
    if abnormal_links(features['hrefs']):
        score +=1; reasons.append("Suspicious links found")
    if forms_action_abnormal(features['forms'], host):
        score +=2; reasons.append("Suspicious form actions")
    if anchors_point_elsewhere(features['hrefs'], host):
        score +=1; reasons.append("Many anchors point elsewhere")
    if meta_keyword_mismatch(features['meta_keywords'], host):
        score +=1; reasons.append("Meta keywords mismatch")

    # URL characteristics
    dcount = digit_count(url)
    ulen = url_length(url)
    uentropy = url_entropy(url)

    if dcount > 5:
        score += 1; reasons.append(f"Too many digits ({dcount})")
    if ulen > 75:
        score += 1; reasons.append(f"URL too long ({ulen} chars)")
    if uentropy > 4.0:
        score += 1; reasons.append(f"High URL entropy ({uentropy:.2f})")

    features.update({"digit_count": dcount, "url_length": ulen, "url_entropy": uentropy})
    return score, reasons, features, host, scheme
