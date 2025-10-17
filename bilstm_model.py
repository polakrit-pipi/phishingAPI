# bilstm_model.py
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

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
# BiLSTMModel
# -----------------------------
class BiLSTMModel:
    def __init__(self, model_path, tokenizer_path, scaler_path, labelencoder_path):
        try:
            self.model = load_model(model_path, custom_objects={"Attention": Attention})
            print("✅ BiLSTM Model with Attention loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load BiLSTM model: {e}")
            self.model = None

        try:
            self.tokenizer = joblib.load(tokenizer_path)
            self.scaler = joblib.load(scaler_path)
            self.labelencoder = joblib.load(labelencoder_path)
            print("✅ Tokenizer, Scaler, LabelEncoder loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load preprocessing objects: {e}")
            self.tokenizer, self.scaler, self.labelencoder = None, None, None

    def preprocess(self, url: str, maxlen: int = 100):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        seq = self.tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=maxlen)
        return padded

    def predict(self, url: str):
        # ถ้า model หรือ tokenizer/scaler/labelencoder หาย ให้ fallback
        if self.model is None or self.tokenizer is None:
            return self.fallback_predict(url)

        X = self.preprocess(url)
        X_scaled = self.scaler.transform(X) if hasattr(self.scaler, "transform") else X
        probs = self.model.predict(X_scaled, verbose=0)
        label_index = np.argmax(probs, axis=1)[0]
        label = self.labelencoder.inverse_transform([label_index])[0]
        return label, float(np.max(probs))

    def fallback_predict(self, url: str):
        # Simple fallback probabilities
        safe_keywords = ['google', 'facebook', 'amazon', 'microsoft', 'github', 'official']
        suspicious_keywords = ['paypal', 'login', 'verify', 'banking', 'secure', 'account', 'password']

        url_lower = url.lower()
        if any(s in url_lower for s in safe_keywords):
            probs = [0.8, 0.2]
            label = "safe"
        elif any(s in url_lower for s in suspicious_keywords):
            probs = [0.3, 0.7]
            label = "phishing"
        else:
            probs = [0.6, 0.4]
            label = "safe"

        return label, float(np.max(probs))
