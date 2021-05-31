#!/usr/bin/env python
import argparse
from review_log_classifier import review_log_classifier
from nb_classifier import nb_classifier
from random_forest_classifier import random_forest_classifier
from support_vector_classifier import support_vector_classifier
import utils
import sklearn
from sklearn.metrics import accuracy_score

""" main method for classifier instantiation and classification """


def main(args: argparse.Namespace) -> None:
    df = utils.extract_features(args.train)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        df["review"], df["num_sentiment"]
    )
    classifier = args.model
    if args.model == "support_vector ":
        classifier = support_vector_classifier()
    elif args.model == "logistic_regression":
        classifier = review_log_classifier()
    elif args.model == "random_forest ":
        classifier = random_forest_classifier()
    else:
        classifier = nb_classifier()
    classifier.train(X_train, y_train)
    preds = classifier.predict(X_test)
    score = accuracy_score(preds, y_test)
    print(f"score: ", score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", help="path to input train TSV")
    parser.add_argument("model", help="output model")
    main(parser.parse_args())
