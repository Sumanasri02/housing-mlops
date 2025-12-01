import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

INPUT_PATH = "data/ingested_data.csv"
OUTPUT_PATH = "data/preprocessed_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def main():
    try:
        df = pd.read_csv(INPUT_PATH)

        target = "price"
        X = df.drop(columns=[target])
        y = df[target]

        numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns

        numerical_pipeline = Pipeline([("scaler", StandardScaler())])
        categorical_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preprocessor.fit(X_train)

        transformed = preprocessor.transform(X)
        transformed_df = pd.DataFrame(transformed.toarray() if hasattr(transformed, "toarray") else transformed)
        transformed_df[target] = y.values

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        transformed_df.to_csv(OUTPUT_PATH, index=False)

        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        print("🔄 Data preprocessing complete")
        return True

    except Exception as e:
        print("❌ Preprocessing failed:", e)
        return False
