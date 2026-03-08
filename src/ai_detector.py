import joblib
import os

MODEL_PATH = "models/fatigue_model.pkl"

if os.path.exists(MODEL_PATH):

    model = joblib.load(MODEL_PATH)

else:

    model = None


def predict_fatigue(ear, mar, blink_rate, head_tilt, sleep_duration):

    if model is None:

        if ear < 0.25 or sleep_duration > 2:
            return "DROWSY", 80

        return "SAFE", 20

    features = [[ear, mar, blink_rate, head_tilt, sleep_duration]]

    pred = model.predict(features)[0]

    if pred == 1:
        return "DROWSY", 85

    return "SAFE", 20