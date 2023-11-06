from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.exceptions import InvalidSentenceError
from app.api.models import PredictionRequest, PredictionResponse
from app.inference.masking import get_masked_language_model

app = FastAPI()


@app.post("/api/predict-word", response_model=PredictionResponse)
def predict_word(request: PredictionRequest):
    model = get_masked_language_model()
    words = model.predict_word(request.sentences)
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
