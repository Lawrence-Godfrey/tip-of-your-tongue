from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    sentences: List[str]
    n: int = 5


class PredictionResponse(BaseModel):
    words: List[tuple[str, float]]
