from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .exceptions import InvalidSentenceError
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


@app.exception_handler(InvalidSentenceError)
def handle_invalid_sentence_error(request, exc: InvalidSentenceError):
    return JSONResponse(
        status_code=400,
        content={
            "sentence": exc.sentence,
            "message": str(exc),
        }
    )
