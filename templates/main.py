from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from fastapi.templating import Jinja2Templates

# Load model
model = joblib.load("linear.pkl")

# App
app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, hours: float = Form(...)):
    # Prepare data
    features = np.array([[hours]])
    prediction = model.predict(features)[0]
    
    return templates.TemplateResponse("result.html", {"request": request, "hours": hours, "prediction": prediction})

