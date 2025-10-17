# model_utils.py
import os, joblib, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# -----------------------------
# Configuration
# -----------------------------
maxlen = 200

# -----------------------------
# Custom Attention Layer
# -----------------------------
@register_keras_serializable()
class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a_it = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        return tf.reduce_sum(x * a_it, axis=1)

# -----------------------------
# Fallback Model
# -----------------------------
class FallbackModel:
    def __init__(self):
        self.pattern_probs = {
            # Safe domains
            'google.com': [0.85, 0.15],
            'facebook.com': [0.82, 0.18],
            'github.com': [0.88, 0.12],
            'amazon.com': [0.80, 0.20],
            'microsoft.com': [0.83, 0.17],
            'example.com': [0.75, 0.25],
            # Suspicious patterns
            'paypal': [0.25, 0.75],
            'login': [0.35, 0.65],
            'verify': [0.30, 0.70],
            'banking': [0.28, 0.72],
            'secure': [0.32, 0.68],
            # Default
            'default': [0.60, 0.40]
        }

    def predict_proper(self, url):
        url_lower = url.lower()
        for pattern, prob in self.pattern_probs.items():
            if pattern in url_lower and pattern != 'default':
                return np.array([prob])
        suspicious_keywords = ['paypal','login','verify','banking','secure','account','password']
        safe_keywords = ['google','facebook','amazon','microsoft','github','official']
        if any(s in url_lower for s in suspicious_keywords):
            return np.array([[0.3,0.7]])
        elif any(s in url_lower for s in safe_keywords):
            return np.array([[0.8,0.2]])
        return np.array([[0.6,0.4]])

    def predict(self, x):
        return np.array([[0.5,0.5]])  # compatibility

# -----------------------------
# Load REAL models
# -----------------------------
scaler, tokenizer, le, model = None, None, None, None
model_files = {
    'scaler': '/content/drive/MyDrive/Project/AJJOKE/project/utils/scaler-2.joblib',
    'tokenizer': '/content/drive/MyDrive/Project/AJJOKE/project/utils/tokenizer-2.joblib',
    'labelencoder': '/content/drive/MyDrive/Project/AJJOKE/project/utils/labelencoder-2.joblib',
    'model': '/content/drive/MyDrive/Project/AJJOKE/project/utils/model.keras'
}

print("🔍 Loading REAL models from Google Drive...")
try:
    if os.path.exists(model_files['scaler']):
        scaler = joblib.load(model_files['scaler'])
    if os.path.exists(model_files['tokenizer']):
        tokenizer = joblib.load(model_files['tokenizer'])
    if os.path.exists(model_files['labelencoder']):
        le = joblib.load(model_files['labelencoder'])
    if os.path.exists(model_files['model']):
        model = load_model(model_files['model'], custom_objects={"Attention": Attention})
except Exception as e:
    print(f"⚠️ Error loading model files: {e}")

if all([scaler, tokenizer, le, model]):
    print("🎉 SUCCESS: All REAL models loaded!")
else:
    print("⚠️ Missing components. Using fallback model.")
    model = FallbackModel()
