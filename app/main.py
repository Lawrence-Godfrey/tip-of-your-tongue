from fastapi import FastAPI
from .models import PredictionRequest, PredictionResponse
from .inference import aggregate_predictions

app = FastAPI()


@app.post("/api/predict-word", response_model=PredictionResponse)
def predict_word(request: PredictionRequest):
    words = aggregate_predictions(request.sentences)
    return {
        "sentences": request.sentences,
        "words": words,
    }
