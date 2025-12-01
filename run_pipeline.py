from src.data_ingestion import main as ingest
from src.preprocessing.data_preprocessing import main as preprocess
from src.model_trainer.model_trainer import main as train
from src.evaluation.evaluate_model import main as evaluate

if __name__ == "__main__":
    print("\n🚀 Starting ML Pipeline...\n")

    print("📥 Step 1: Data Ingestion")
    ingest()

    print("🔄 Step 2: Data Preprocessing")
    preprocess()

    print("🧠 Step 3: Model Training")
    train()

    print("📊 Step 4: Model Evaluation")
    evaluate()

    print("\n🎉 Pipeline Completed Successfully!")
