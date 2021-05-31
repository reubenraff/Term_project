#!/usr/bin/env python
"""MultinomialNB classifier for pet names."""


from typing import Any, Dict, List
import numpy as np
import pandas as pd
import sklearn.feature_extraction  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn  # type: ignore
import utils


class nb_classifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        # self.tf_vectorizer = TfidfVectorizer()
        self.cnt_vectorizer = CountVectorizer()

        # I use a count vectorizer for the naive bayes feature representation
        # and a multinomial Naive bayes model instance
        self.classifier = MultinomialNB()

    def train(self, xx, y):
        xx = self.cnt_vectorizer.fit_transform(np.array(xx))
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.cnt_vectorizer.transform(np.array(x))
        return list(self.classifier.predict(xx))
