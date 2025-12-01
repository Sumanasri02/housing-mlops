# 🏡 Housing Price Prediction — MLOps Project

An end-to-end **Machine Learning + MLOps** project that predicts house prices using the California Housing dataset.  
This project demonstrates production-grade architecture including model training, CI automation, Docker deployment, and cloud hosting on Render.

---

## 🚀 Project Workflow

| Stage | Technology Used | Description |
|------|----------------|-------------|
| Data Ingestion | Pandas, Scikit-learn | Load and preprocess the housing dataset |
| Model Training | Random Forest Regression | Train & evaluate model metrics |
| Packaging | Docker | Create containerized ML application |
| CI Pipeline | GitHub Actions | Linting, dependency installation, Docker build check |
| Deployment | Render Cloud | Host application online |

---

## 📂 Project Structure
housing-mlops/
│
├── api/
│ ├── app.py # FastAPI app for UI & API
│ ├── predict.py # Prediction logic
│ └── init.py
│
├── templates/
│ └── index.html # UI Page for user input
│
├── models/
│ └── model.pkl # Trained model artifact
│
├── Dockerfile # Docker configuration
├── requirements.txt # Dependencies
└── .github/workflows/ci.yml # CI automation pipeline
housing-mlops/
│
├── api/
│ ├── app.py # FastAPI app for UI & API
│ ├── predict.py # Prediction logic
│ └── init.py
│
├── templates/
│ └── index.html # UI Page for user input
│
├── models/
│ └── model.pkl # Trained model artifact
│
├── Dockerfile # Docker configuration
├── requirements.txt # Dependencies
└── .github/workflows/ci.yml # CI automation pipeline 



---

## 🔧 How to Run Locally

```bash
git clone https://github.com/Sumanasri02/housing-mlops.git
cd housing-mlops
pip install -r requirements.txt
python api/app.py
🐳 Run with Docker
docker build -t housing-app .
docker run -p 5000:5000 housing-app
🌍 Live Hosted Application
🔗 https://housing-mlops.onrender.com/
🧪 CI Pipeline — GitHub Actions

✔️ Install dependencies
✔️ Linting check (flake8)
✔️ Docker validation build
✔️ Status badge coming soon!

🧠 Model

Algorithm: Random Forest Regressor

Advanced modeling can be added later with hyperparameter tuning + MLflow tracking

🙌 Developer

👩‍💻 Sumanasri
Passionate about ML Deployment & MLOps 🚀
