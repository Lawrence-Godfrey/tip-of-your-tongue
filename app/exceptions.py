
class InvalidSentenceError(Exception):
    def __init__(self, message: str, sentence: str):
        self.sentence = sentence
        super().__init__(message)
