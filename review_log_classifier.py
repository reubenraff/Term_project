#!/usr/bin/env python

"""LogisticRegression classifier for sentiment analysis."""


from typing import Any, Dict, List
import numpy as np
import pandas as pd
import sklearn.feature_extraction  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn  # type: ignore
import utils
from utils import train_test_split


class review_log_classifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=1000
        )
        # instances of a tfidf vectorizer and a logisitic regression
        # model
        self.classifier = sklearn.linear_model.LogisticRegression()

        # I convert this to a string because the vectorizer expects string-like
        # features.

    def train(self, x: List[str], y: List[str]):
        xx = self.vectorizer.fit_transform(x)
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.vectorizer.fit_transform(np.array(x))
        return list(self.classifier.predict(xx))
