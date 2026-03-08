import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Loading dataset...")

df = pd.read_csv("data/fatigue_training.csv")

X = df[["EAR", "MAR", "BLINK_RATE", "HEAD_TILT"]]
y = df["Fatigue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Model Accuracy:", score)

pickle.dump(model, open("models/fatigue_model.pkl", "wb"))

print("Model saved!")