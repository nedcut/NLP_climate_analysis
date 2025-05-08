import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(2,3))
        self.model = MultinomialNB()

    def train(self, texts, labels):
        """
        Trains the Naive Bayes model.

        Args:
            texts (list of str): The input texts.
            labels (list of int): The corresponding labels.
        """
        X = self.vectorizer.fit_transform(texts)
        # Fit with optional sample weights
        if hasattr(self, 'sample_weights') and self.sample_weights is not None:
            self.model.fit(X, labels, sample_weight=self.sample_weights)
        else:
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
