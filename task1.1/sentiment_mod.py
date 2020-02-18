import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



#class of Classifier to classify sentence
class Classifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    #classify function
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # function to predict how many percent
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


#open document that had been pickled
documents_f = open("pickle/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


#open word feature that had been pickled
word_features5k_f = open("pickle/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#open featuresets that had been pickled
featuresets_f = open("pickle/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

# order the datasets
random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


#open originial naive bayes classifier that had been pickled to reused
open_file = open("pickle/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


#open multinomial Naive Bayes classifier that had been pickled to reused
open_file = open("pickle/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

#open bernoulli Naive Bayes classifier that had been pickled to reused
open_file = open("pickle/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

#open Logistic Regression classifier that had been pickled to reused
open_file = open("pickle/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


#open Linear SVC classifier that had been pickled to reused
open_file = open("pickle/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


#open Stocastic Gradient Descent classifier that had been pickled to reused
open_file = open("pickle/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


classifier = Classifier(
                    classifier,
                    LinearSVC_classifier,
                    MNB_classifier,
                    BernoulliNB_classifier,
                    LogisticRegression_classifier
                )


# function to determine and classfy sentiment from sentence that given
def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats),classifier.confidence(feats)
