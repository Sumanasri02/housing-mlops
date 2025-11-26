import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def preprocessing_pipeline(df):
    #1 identify features
    target = "price"
    X = df.drop(columns=[target])
    y = df[target]
    #seperate columns types
    numerical_cols = X.select_dtypes(include=['int64','float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    #pipeline
    numerical_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    #column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )
    
    #3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return preprocessor, X_train,X_test,y_train, y_test

if __name__ == "__main__":
     # 1. Load dataset
    df = pd.read_csv(r"D:\Techflitter\housing-mlops\data\Housing.csv")

    # 2. Get pipeline and splits
    preprocessor, X_train, X_test, y_train, y_test = preprocessing_pipeline(df)
    # 3. Fit preprocessor on TRAIN data only
    preprocessor.fit(X_train)

    # 4. Transform full dataset for model training
    full_transformed = preprocessor.transform(df.drop(columns=["price"]))

    # 5. Convert transformed data into dataframe
    # Get encoded column names
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns
    cat_cols = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out()

    all_feature_names = list(num_cols) + list(cat_cols)

    transformed_df = pd.DataFrame(full_transformed, columns=all_feature_names)
    transformed_df["price"] = df["price"]

    # 6. Save final CSV
    output_path = r"D:\Techflitter\housing-mlops\data\preprocessed_data.csv"
    transformed_df.to_csv(output_path, index=False)

    print("Saved preprocessed_data.csv")

    # 7. Save preprocessor.pkl
    joblib.dump(preprocessor, r"D:\Techflitter\housing-mlops\models\preprocessor.pkl")

    print("Saved preprocessor.pkl")