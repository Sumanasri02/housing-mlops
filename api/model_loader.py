import joblib
import os

MODEL_PATH = os.path.join("models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def load_preprocessor():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return preprocessor
