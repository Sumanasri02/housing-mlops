import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

DATA_PATH = "/opt/airflow/data/ingested_data.csv"
REPORTS_DIR = "/opt/airflow/reports"

def run_eda():
    df = pd.read_csv(DATA_PATH)

    os.makedirs(REPORTS_DIR + "/numeric_distributions", exist_ok=True)
    os.makedirs(REPORTS_DIR + "/numeric_boxplots", exist_ok=True)
    os.makedirs(REPORTS_DIR + "/categorical_plots", exist_ok=True)
    os.makedirs(REPORTS_DIR + "/correlation", exist_ok=True)
    os.makedirs(REPORTS_DIR + "/cat_vs_target", exist_ok=True)
    os.makedirs(REPORTS_DIR + "/target", exist_ok=True)

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=False)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{REPORTS_DIR}/numeric_distributions/{col}_distribution.png")
        plt.close()

    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{REPORTS_DIR}/numeric_boxplots/{col}_boxplot.png")
        plt.close()

    plt.figure(figsize=(12,8))
    corr_matrix = df[numeric_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig(f"{REPORTS_DIR}/correlation/correlation_heatmap.png")
    plt.close()

    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        plt.figure(figsize=(6,4))
        sns.countplot(x=df[col])
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.savefig(f"{REPORTS_DIR}/categorical_plots/{col}_countplot.png")
        plt.close()

    target = "price"
    for col in categorical_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.savefig(f"{REPORTS_DIR}/cat_vs_target/{col}_vs_price.png")
        plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(df[target], kde=True)
    plt.title("Distribution of Price")
    plt.savefig(f"{REPORTS_DIR}/target/price_distribution.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[target])
    plt.title("Boxplot of Price")
    plt.savefig(f"{REPORTS_DIR}/target/price_boxplot.png")
    plt.close()

    df["price_log"] = np.log(df[target].replace(0, np.nan).dropna())
    plt.figure(figsize=(6,4))
    sns.histplot(df["price_log"].dropna(), kde=True)
    plt.title("Log-Transformed Price Distribution")
    plt.savefig(f"{REPORTS_DIR}/target/price_log_distribution.png")
    plt.close()

    print("EDA Completed Successfully.")


if __name__ == "__main__":
    run_eda()
