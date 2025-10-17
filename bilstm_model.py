import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BiLSTMModel:
    def __init__(self, model_path, tokenizer_path, scaler_path, labelencoder_path):
        self.model = load_model(model_path)

        # โหลดไฟล์ด้วย joblib
        self.tokenizer = joblib.load(tokenizer_path)
        self.scaler = joblib.load(scaler_path)
        self.labelencoder = joblib.load(labelencoder_path)

    def preprocess(self, url: str):
        seq = self.tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=100)
        return padded

    def predict(self, url: str):
        X = self.preprocess(url)
        scaled = self.scaler.transform(X) if hasattr(self.scaler, "transform") else X
        probs = self.model.predict(scaled, verbose=0)
        label_index = np.argmax(probs, axis=1)[0]
        label = self.labelencoder.inverse_transform([label_index])[0]
        return label, float(np.max(probs))
