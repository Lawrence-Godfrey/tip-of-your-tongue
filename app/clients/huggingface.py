from typing import List, Union
from urllib.parse import urljoin

import requests


class HuggingFaceClient:
    """
    Client for interacting with the HuggingFace inference API.
    """

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

        self.endpoint = urljoin(self.api_url, self.model_name)

    def get_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def predict_word(
            self, sentence: Union[str, List[str]]
    ) -> Union[List[dict], List[List[dict]]]:
        """
        Predict the most likely word to fill in the blank for each sentence.

        Args:
            sentence: A sentence or list of sentences with a blank in it, represented by '[MASK]'.

        Returns:
            A list ( or list of lists) of tuples containing the predicted word and its probability.
        """

        response = requests.post(self.endpoint, headers=self.get_headers(), json={"inputs": sentence})

        if response.status_code != 200:
            # Try and get the error message from the response
            try:
                message = response.json()["error"]
            except KeyError:
                message = "An unknown error occurred"

            raise ValueError(f"Failed to get predictions: {message}")

        return response.json()
