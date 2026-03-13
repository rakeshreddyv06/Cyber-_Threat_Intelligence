from tensorflow.keras.models import load_model
import joblib
import numpy as np

from src.feature_extraction import extract_features
from src.sequence_buffer import build_sequence

model = load_model("model/convlstm_model.h5")

scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")


def detect(packet):

    features = extract_features(packet)

    if features is None:
        return None

    features = scaler.transform([features])[0]

    sequence = build_sequence(features)

    if sequence is None:
        return None

    prediction = model.predict(sequence)

    attack_index = np.argmax(prediction)

    attack_type = encoder.inverse_transform([attack_index])[0]

    confidence = float(np.max(prediction)) * 100

    try:
        src = packet["IP"].src
        dst = packet["IP"].dst
    except:
        src = "Unknown"
        dst = "Unknown"

    return {
        "source": src,
        "destination": dst,
        "attack_type": attack_type,
        "confidence": round(confidence,2)
    }