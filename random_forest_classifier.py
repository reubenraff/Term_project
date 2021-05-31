#!/usr/bin/env python
"""RandomForest classifier for pet names."""


from typing import Any, Dict, List

import sklearn.feature_extraction  # type: ignore
import sklearn  # type: ignore
from sklearn import ensemble


class random_forest_classifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        # Instantiation of a tfidf vectorizer and a random
        # forest classifier
        self.classifier = sklearn.ensemble.RandomForestClassifier()

    def train(self, x: List[str], y: List[str]):
        xx = self.vectorizer.fit_transform(x)
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.vectorizer.transform(x)
        return list(self.classifier.predict(xx))
