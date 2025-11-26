import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def run_eda():

    # -------------------------
    # 1. Load the dataset
    # -------------------------
    data_path = r"D:\Techflitter\housing-mlops\data\Housing.csv"
    df = pd.read_csv(data_path)

    print("First 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    # -------------------------
    # 2. Missing Values
    # -------------------------
    print("\nMissing Values:")
    print(df.isnull().sum())

    # -------------------------
    # 3. Numeric Features
    # -------------------------
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nNumeric Columns:", numeric_features)
    print(df[numeric_features].describe())

    os.makedirs(r"D:\Techflitter\housing-mlops\reports\numeric_distributions", exist_ok=True)

    # Histograms
    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=False)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"D:\\Techflitter\\housing-mlops\\reports\\numeric_distributions\\{col}_distribution.png")
        plt.close()

    # Boxplots
    os.makedirs(r"D:\Techflitter\housing-mlops\reports\numeric_boxplots", exist_ok=True)
    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"D:\\Techflitter\\housing-mlops\\reports\\numeric_boxplots\\{col}_boxplot.png")
        plt.close()

    # -------------------------
    # 4. Correlation Heatmap
    # -------------------------
    plt.figure(figsize=(12,8))
    corr_matrix = df[numeric_features].corr()

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")

    os.makedirs(r"D:\Techflitter\housing-mlops\reports\correlation", exist_ok=True)
    plt.savefig(r"D:\Techflitter\housing-mlops\reports\correlation\correlation_heatmap.png")
    plt.close()

    # -------------------------
    # 5. Categorical Features
    # -------------------------
    categorical_features = df.select_dtypes(include=['object']).columns
    print("\nCategorical columns:", categorical_features)

    for col in categorical_features:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

    os.makedirs(r"D:\Techflitter\housing-mlops\reports\categorical_plots", exist_ok=True)

    for col in categorical_features:
        plt.figure(figsize=(6,4))
        sns.countplot(x=df[col])
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.savefig(f"D:\\Techflitter\\housing-mlops\\reports\\categorical_plots\\{col}_countplot.png")
        plt.close()

    # -------------------------
    # 6. Categorical vs Target
    # -------------------------
    target = "price"
    os.makedirs(r"D:\Techflitter\housing-mlops\reports\cat_vs_target", exist_ok=True)

    for col in categorical_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.savefig(f"D:\\Techflitter\\housing-mlops\\reports\\cat_vs_target\\{col}_vs_price.png")
        plt.close()

    # -------------------------
    # 7. Target Variable Analysis
    # -------------------------
    os.makedirs(r"D:\Techflitter\housing-mlops\reports\target", exist_ok=True)

    # Distribution
    plt.figure(figsize=(6,4))
    sns.histplot(df[target], kde=True)
    plt.title("Distribution of Price")
    plt.savefig(r"D:\Techflitter\housing-mlops\reports\target\price_distribution.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[target])
    plt.title("Boxplot of Price")
    plt.savefig(r"D:\Techflitter\housing-mlops\reports\target\price_boxplot.png")
    plt.close()

    # Skewness
    print("\nSkewness of price:", df[target].skew())

    # Log transform
    df["price_log"] = np.log(df[target])
    plt.figure(figsize=(6,4))
    sns.histplot(df["price_log"], kde=True)
    plt.title("Log-Transformed Price Distribution")
    plt.savefig(r"D:\Techflitter\housing-mlops\reports\target\price_log_distribution.png")
    plt.close()

    print("\nEDA Completed Successfully.")



if __name__ == "__main__":
    run_eda()
