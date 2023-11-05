from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import torch


# Load model and tokenizer
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def predict_with_scores(sentence: str, n: int = 5) -> list[tuple[str, float]]:
    """
    Predict the word that best fits the blank in the sentence.

    Args:
        sentence: A sentence with a blank in it, represented by '____'.
        n: The number of predictions to return.

    Returns:
        A list of tuples containing the predicted word and its score.
    """
    # Replace blank with <mask>
    sentence = sentence.replace("____", "[MASK]")

    # Tokenize the sentence
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]

    # Get model predictions
    with torch.no_grad():
        prediction = model(input_ids).logits

    # Get predicted token IDs and their scores
    masked_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    top_n_token_ids = prediction[0, masked_index].topk(n).indices
    top_n_token_ids = top_n_token_ids.squeeze()

    top_n_scores = prediction[0, masked_index].topk(n).values
    top_n_scores = top_n_scores.squeeze()

    # Decode token IDs to words
    top_n_words = [tokenizer.decode(token_id).replace(' ', '') for token_id in top_n_token_ids]

    return list(zip(top_n_words, top_n_scores.tolist()))


def aggregate_predictions(sentences: list, n: int = 5) -> list[tuple[str, float]]:
    """
    Aggregate the predictions for a list of sentences.

    Args:
        sentences: A list of sentences with a blank in it, represented by '____'.
        n: The number of predictions to return.

    Returns:
        A list of tuples containing the predicted word and its score.
    """
    word_scores = {}

    for sentence in sentences:
        predictions = predict_with_scores(sentence, n * len(sentences))
        predictions = normalise_predictions(predictions)
        for word, score in predictions:
            if word in word_scores:
                word_scores[word] += score
            else:
                word_scores[word] = score

    # Return the first n words with the highest scores
    return sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]


def normalise_predictions(predictions: list[tuple[str, float]]) -> list[tuple[str, float]]:
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
