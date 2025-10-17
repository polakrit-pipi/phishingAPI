# predict_utils.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_utils import model, scaler, tokenizer, le, maxlen
from feature_utils import parse_host_and_scheme, is_ip_host, count_subdomains, has_double_slash_in_path, has_tld_in_path, has_symbols_in_domain, domain_prefix_suffix_like_brand, brand_in_path_or_subdomain, digit_count, url_entropy

def predict_url(url):
    host, scheme = parse_host_and_scheme(url)
    if hasattr(model, 'last_url'):
        model.last_url = url

    # fallback mode
    if scaler is None or tokenizer is None or le is None:
        pred = model.predict([None])[0]
        label = "Likely Safe" if pred[0] > pred[1] else "Likely Phishing"
        return label, pred

    try:
        struct_feat = scaler.transform([[
            int(is_ip_host(host)),
            count_subdomains(host),
            int(has_double_slash_in_path(url)),
            int(has_tld_in_path(url)),
            int(has_symbols_in_domain(host)),
            int(domain_prefix_suffix_like_brand(host)),
            int(brand_in_path_or_subdomain(host, url)),
            len(url),
            1 if scheme == 'https' else 0,
            digit_count(url),
            url_entropy(url)
        ]])

        seq = pad_sequences(tokenizer.texts_to_sequences([url]), maxlen=maxlen)
        pred = model.predict([seq, struct_feat])[0]
        label = le.inverse_transform([np.argmax(pred)])[0]
        return label, pred

    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        pred = model.predict([None])[0]
        label = "Likely Safe" if pred[0] > pred[1] else "Likely Phishing"
        return label, pred
