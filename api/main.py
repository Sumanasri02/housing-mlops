from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from pathlib import Path 

# Define the base path as the directory where the main.py file is located
BASE_DIR = Path(__file__).resolve().parent

# Load model & preprocessor from the PARENT directory's 'models' folder
MODEL_PATH = BASE_DIR.parent / "models/best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR.parent / "models/preprocessor.pkl"
TEMPLATES_DIR = BASE_DIR.parent / "templates"


# Load the actual model files using the constructed paths
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print(f"Please ensure your files are in the correct location: {MODEL_PATH.parent}")
    raise SystemExit("Exiting due to missing model files.") # Stop the app if files are missing

app = FastAPI()
# Pass the Path object to Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR) 
Instrumentator().instrument(app).expose(app)

class HouseInput(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_price(data: HouseInput):
    input_df = pd.DataFrame([data.dict()])
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)
    # Ensure prediction is converted to a float for the JSON response
    return {"predicted_price": float(prediction)}
