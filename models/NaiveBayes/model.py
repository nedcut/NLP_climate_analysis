\
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, texts, labels):
        """
        Trains the Naive Bayes model.

        Args:
            texts (list of str): The input texts.
            labels (list of int): The corresponding labels.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts):
        """
        Predicts labels for the given texts.

        Args:
            texts (list of str): The input texts.

        Returns:
            np.ndarray: The predicted labels.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """
        Predicts class probabilities for the given texts.

        Args:
            texts (list of str): The input texts.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

if __name__ == '__main__':
    # Example Usage (optional - for testing)
    texts_train = ["this is a positive example", "another positive one", "negative sentiment here", "this is bad"]
    labels_train = [1, 1, 0, 0] # 1 for positive, 0 for negative

    texts_test = ["this is a good one", "this is a terrible one"]

    nb_model = NaiveBayesModel()
    nb_model.train(texts_train, labels_train)

    predictions = nb_model.predict(texts_test)
    print(f"Predictions: {predictions}") # Expected: [1 0] or similar depending on tokenization

    probabilities = nb_model.predict_proba(texts_test)
    print(f"Probabilities: {probabilities}")
