import boto3
import json
import requests

from abc import abstractmethod
from typing import List, Union
from urllib.parse import urljoin

from app.clients.huggingface import HuggingFaceClient
from gradio_client import Client


def get_masked_language_model():
    from dotenv import load_dotenv
    import os

    load_dotenv()

    if os.getenv('MODEL_TYPE') == 'huggingface-api':
        return HuggingFaceModel(
            os.getenv('HUGGINGFACE_API_URL'),
            os.getenv('HUGGINGFACE_API_KEY'),
            os.getenv('MODEL_NAME')
        )
    elif os.getenv('MODEL_TYPE') == 'sagemaker':
        return SageMakerModel(os.getenv('SAGEMAKER_ENDPOINT_NAME'))

    elif os.getenv('MODEL_TYPE') == 'gradio':
        return GradioModel(os.getenv('GRADIO_URL'))

    elif os.getenv('MODEL_TYPE') == 'local':
        return LocalMaskingModel().get_model_from_name(os.getenv('MODEL_NAME'))
    else:
        raise ValueError('Invalid model type')


class BaseMasking:

    @abstractmethod
    def predict_word(self, sentence: Union[str, List[str]], n: int = 5) -> List[tuple[str, float]]:
        """
        Predict the most likely word to fill in the blank for each sentence.

        Args:
            sentence: A sentence or list of sentences with a blank in it, represented by '[MASK]'.
            n: The number of predictions to return.

        Returns:
            A list of tuples containing the predicted word and its probability.
        """
        pass

    @abstractmethod
    def normalise_sentence(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Normalise the sentence to be used by the model.

        Args:
            sentence: A sentence or list of sentences with a blank in it, represented by '[MASK]'.

        Returns:
            A normalised sentence.
        """
        pass

    def aggregate_predictions(
            self, predictions: Union[List[tuple[str, float]], List[List[tuple[str, float]]]], n: int = 5
    ) -> List[tuple[str, float]]:
        """
        Aggregate the predictions into a list of tuples containing the predicted word and its probability.

        Args:
            predictions: A list of predictions, where each prediction is a list of tuples containing the predicted
                word and its probability.
            n: The number of predictions to return.

        Returns:
            A list of tuples containing the predicted word and its probability.
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

    def normalise_predictions(self, predictions):
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


class HuggingFaceModel(BaseMasking):

    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    def predict_word(self, sentence, n=5):
        hf_client = HuggingFaceClient(self.api_url, self.api_key, self.model_name)
        predictions = hf_client.predict_word(self.normalise_sentence(sentence))
        return self.aggregate_predictions(predictions, n)

    def normalise_sentence(self, sentence):
        if isinstance(sentence, list):
            return [s.replace('____', '[MASK]') for s in sentence]
        else:
            return sentence.replace('____', '[MASK]')

    def aggregate_predictions(self, predictions: Union[List[dict], List[List[dict]]], n=5):
        if not isinstance(predictions[0], list):
            predictions = [predictions]

        predictions = [[(p['token_str'], p['score']) for p in prediction] for prediction in predictions]

        return super().aggregate_predictions(predictions, n)


class SageMakerModel(BaseMasking):
    """
    This class interfaces with models deployed on AWS SageMaker.
    """

    def __init__(self, endpoint_name: str):
        """
        Args:
            endpoint_name: The name of the SageMaker endpoint.
        """
        self.endpoint_name = endpoint_name

        self.runtime = boto3.client('sagemaker-runtime')

    def predict_word(self, sentence, n=5):
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"inputs": self.normalise_sentence(sentence)}),
        )

        predictions = json.loads(response['Body'].read().decode())

        return self.aggregate_predictions(predictions, n)

    def normalise_sentence(self, sentence):
        if isinstance(sentence, list):
            return [s.replace('____', '[MASK]') for s in sentence]
        else:
            return sentence.replace('____', '[MASK]')

    def aggregate_predictions(self, predictions: Union[List[dict], List[List[dict]]], n=5):
        if not isinstance(predictions[0], list):
            predictions = [predictions]

        predictions = [[(p['token_str'], p['score']) for p in prediction] for prediction in predictions]

        return super().aggregate_predictions(predictions, n)


class GradioModel(BaseMasking):
    """
    This class interfaces with models deployed on Gradio.
    """

    def __init__(self, url: str):
        """
        Args:
            url: The URL of the Gradio model. E.g. <username>/google-bert-bert-base-uncased.
        """
        self.url = url

    def predict_word(self, sentences: Union[str, List[str]], n: int = 5) -> List[tuple[str, float]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        client = Client(self.url)

        results = []

        for sentence in sentences:
            results.extend(client.predict(self.normalise_sentence(sentence), api_name='/predict')['confidences'])

        return self.aggregate_predictions(results, n)

    def normalise_sentence(self, sentence):
        if isinstance(sentence, list):
            return [s.replace('____', '[MASK]') for s in sentence]
        else:
            return [sentence.replace('____', '[MASK]')]

    def aggregate_predictions(self, predictions: Union[List[dict], List[List[dict]]], n=5):
        if not isinstance(predictions[0], list):
            predictions = [predictions]

        predictions = [[(p['label'], p['confidence']) for p in prediction] for prediction in predictions]

        return super().aggregate_predictions(predictions, n)


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
    def predict_word(self, sentence, n=5):
        pass


class DistilBertBaseUncased(BaseMasking):

    def __init__(self):
        super().__init__()

        from transformers import DistilBertForMaskedLM, DistilBertTokenizer

        self.model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def predict_word(self, sentence, n=5):
        import torch

        # Replace blank with <mask>
        sentence = self.normalise_sentence(sentence)

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

    def normalise_sentence(self, sentence):
        if isinstance(sentence, list):
            return [s.replace('____', '[MASK]') for s in sentence]
        else:
            return sentence.replace('____', '[MASK]')


class RobertaBase(LocalMaskingModel):

    def __init__(self):
        super().__init__()

        from transformers import RobertaForMaskedLM, RobertaTokenizer

        self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def predict_word(self, sentence: Union[str, List[str]], n: int = 5) -> List[tuple[str, float]]:

        import torch

        # Replace blank with <mask>
        sentence = self.normalise_sentence(sentence)

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

    def normalise_sentence(self, sentence: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(sentence, list):
            return [s.replace('____', '<mask>') for s in sentence]
        else:
            return sentence.replace('____', '<mask>')
