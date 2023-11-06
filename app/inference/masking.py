from abc import abstractmethod
from typing import List

import torch

from app.clients.huggingface import HuggingFaceClient


def get_masked_language_model():
    from dotenv import load_dotenv
    import os

    load_dotenv()

    if os.getenv('MODEL_TYPE') == 'huggingface-api':
        return HuggingFaceModel(os.getenv('HUGGINGFACE_API_URL'), os.getenv('HUGGINGFACE_API_KEY'))
    elif os.getenv('MODEL_TYPE') == 'local':
        return LocalMaskingModel().get_model_from_name(os.getenv('MODEL_NAME'))
    else:
        raise ValueError('Invalid model type')


class BaseMasking:

    @abstractmethod
    def predict_word(self, sentence: str or List[str], n: int = 5) -> List[tuple[str, float]]:
        """
        Predict the most likely word to fill in the blank for each sentence.

        Args:
            sentence: A sentence or list of sentences with a blank in it, represented by '[MASK]'.
            n: The number of predictions to return.

        Returns:
            A list of tuples containing the predicted word and its probability.
        """
        pass


class HuggingFaceModel(BaseMasking):

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def predict_word(self, sentence: str or List[str], n: int = 5) -> List[tuple[str, float]]:
        return HuggingFaceClient(self.api_url, self.api_key).predict_word(sentence)


class LocalMaskingModel(BaseMasking):

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def get_model_from_name(self, model_name: str):
        if model_name == 'distilbert-base-uncased':
            return DistilBertBaseUncased()

        elif model_name == 'roberta-base':
            return RobertaBase()

        else:
            raise ValueError('Invalid model name')

    @abstractmethod
    def predict_word(self, sentence: str or List[str], n: int = 5) -> List[tuple[str, float]]:
        pass

    def aggregate_predictions(self, predictions: list, n: int = 5) -> list[tuple[str, float]]:
        """
        Aggregate the predictions for a list of sentences.

        Args:
            predictions: A list of lists of tuples containing the predicted word and its score.
            n: The number of predictions to return.

        Returns:
            A list of tuples containing the predicted word and its score.
        """
        word_scores = {}

        for prediction in predictions:
            prediction = self.normalise_predictions(prediction)
            for word, score in prediction:
                if word in word_scores:
                    word_scores[word] += score
                else:
                    word_scores[word] = score

        # Return the first n words with the highest scores
        return sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def normalise_predictions(self, predictions: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Normalise the scores of the predictions.

        Args:
            predictions: A list of tuples containing the predicted word and its score.

        Returns:
            A list of tuples containing the predicted word and its normalised score.
        """
        # Get the max score
        max_score = max([score for _, score in predictions])

        # Normalise the scores
        return [(word, score / max_score) for word, score in predictions]


class DistilBertBaseUncased(BaseMasking):

    def __init__(self):
        from transformers import DistilBertForMaskedLM, DistilBertTokenizer

        self.model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def predict_word(self, sentence: str or List[str], n: int = 5) -> List[tuple[str, float]]:

        # Replace blank with <mask>
        sentence = sentence.replace("____", "[MASK]")

        # Tokenize the sentence
        input_ids = self.tokenizer(sentence, return_tensors="pt")["input_ids"]

        # Get model predictions
        with torch.no_grad():
            prediction = self.model(input_ids).logits

        # Get predicted token IDs and their scores
        masked_index = (input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        top_n_token_ids = prediction[0, masked_index].topk(n).indices
        top_n_token_ids = top_n_token_ids.squeeze()

        top_n_scores = prediction[0, masked_index].topk(n).values
        top_n_scores = top_n_scores.squeeze()

        # Decode token IDs to words
        top_n_words = [self.tokenizer.decode(token_id).replace(' ', '') for token_id in top_n_token_ids]

        return list(zip(top_n_words, top_n_scores.tolist()))


class RobertaBase(LocalMaskingModel):

    def __init__(self):
        from transformers import RobertaForMaskedLM, RobertaTokenizer

        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def predict_word(self, sentence: str or List[str], n: int = 5) -> List[tuple[str, float]]:

        # Replace blank with <mask>

        # Tokenize the sentence
        input_ids = self.tokenizer(sentence, return_tensors="pt")["input_ids"]

        # Get model predictions
        with torch.no_grad():
            prediction = self.model(input_ids).logits

        # Get predicted token IDs and their scores
        masked_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1].item()
        top_n_token_ids = prediction[0, masked_index].topk(n).indices
        top_n_scores = prediction[0, masked_index].topk(n).values

        # Decode token IDs to words
        top_n_words = [self.tokenizer.decode(token_id) for token_id in top_n_token_ids]

        return list(zip(top_n_words, top_n_scores.tolist()))
