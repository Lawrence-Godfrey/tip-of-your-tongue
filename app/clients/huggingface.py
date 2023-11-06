from typing import List

import requests


class HuggingFaceClient:
    """
    Client for interacting with the HuggingFace inference API.
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def predict_word(self, sentence: str or List[str]) -> List[tuple[str, float]]:
        """
        Predict the most likely word to fill in the blank for each sentence.

        Args:
            sentence: A sentence or list of sentences with a blank in it, represented by '[MASK]'.

        Returns:
            A list of tuples containing the predicted word and its probability.
        """

        response = requests.post(self.api_url, headers=self.get_headers(), json={"inputs": sentence})
        return response.json()
