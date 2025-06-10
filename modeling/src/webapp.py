import os
import glob
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from dotenv import load_dotenv
load_dotenv()

import mlflow
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from modeling.src.utils.utils import get_outputs, get_scaler, project_path
from modeling.src.utils.utils import CFG
from modeling.src.inference.inference import inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_root_path = os.path.join(project_path(), 'data')

@app.post("/predict/pm10")
async def predict_pm10(request: Request):
    request_json = await request.json()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = glob.glob(os.path.join(data_root_path, "*_pm10_data.csv"))
    files.sort(key=os.path.getmtime)
    latest_anomalies_file = files[-1]
    data_path = os.path.join(data_root_path, latest_anomalies_file)

    _, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_PM)

    model = mlflow.pytorch.load_model(model_uri="models:/pm10@production")

    fake_test_data = np.random.normal(loc=15, scale=3, size=(CFG['WINDOW_SIZE'], len(outputs_PM)))

    results = inference(model, fake_test_data, scaler, outputs_PM, device)

    return {"prediction": results.tolist()}