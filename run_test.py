#File: sentiment_mod.py

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from nlp import Classifier


documents_f = open("pickle/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features_f = open("pickle/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features5k_f.close()


def find_features(word):
    words = word_tokenize(word)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



featuresets_f = open("pickle/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[3700:]
training_set = featuresets[:3700]



open_file = open("pickle/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickle/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


classifier = Classifier(
                    classifier,
                    LinearSVC_classifier,
                    MNB_classifier,
                    BernoulliNB_classifier,
                    LogisticRegression_classifier
                )

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
