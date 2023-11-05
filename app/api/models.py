from pydantic import BaseModel, field_validator
from typing import List

from app.api.exceptions import InvalidSentenceError


class PredictionRequest(BaseModel):
    sentences: List[str]
    n: int = 5

    @field_validator("sentences", mode="before")
    @classmethod
    def validate_blank_token(cls, sentences):
        for sentence in sentences:
            if "____" not in sentence:
                raise InvalidSentenceError("Sentence does not contain blank token", sentence)

            if sentence.count("____") > 1:
                raise InvalidSentenceError("Sentence contains multiple blank tokens", sentence)

        return sentences


class PredictionResponse(BaseModel):
    words: List[tuple[str, float]]
