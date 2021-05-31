#!/usr/bin/env python
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_features(file):
    # , name: str
    # features: Dict[str, Any] = {}
    df = pd.read_csv(file)
    df["num_sentiment"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    data = df[["review", "num_sentiment"]]
    return data


def split(x, y, test_size, random_state):
    # X = np.array(df["review"])
    # y = np.array(df["sentiment"])
    X_train, X_test, y_train, y_test = train_test_split(
        df.review, df.num_sent, test_size=0.25, random_state=32
    )
    return (X_train, X_test, y_train, y_test)


def score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy
