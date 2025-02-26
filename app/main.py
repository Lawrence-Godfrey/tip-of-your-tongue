
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.logging import logging
from app.api.exceptions import InvalidSentenceError, ModelLoadingError
from app.api.middleware import LoggingMiddleware
from app.api.models import PredictionRequest, PredictionResponse
from app.inference.masking import get_masked_language_model


app = FastAPI()


app.add_middleware(LoggingMiddleware)


origins = [
    "http://localhost:3000",  # The origin of your React client
    "http://127.0.0.1:3000",  # Also include this if you access the site via 127.0.0.1
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


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


@app.exception_handler(ModelLoadingError)
def handle_model_loading_error(request, exc: ModelLoadingError):
    return JSONResponse(
        status_code=400,
        content={
            "message": str(exc),
        }
    )

@app.exception_handler(Exception)
def handle_exception(request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
        }
    )
