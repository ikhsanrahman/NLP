
#1. Import all library needed
import nltk
import random

import datetime
import matplotlib.pyplot as plt
# nltk.download('averaged_perceptron_tagger')

from nltk.classify.scikitlearn import SklearnClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


class Classifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            print(v, 'classify')
            votes.append(v)
            print(votes, 'votes')
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

class PreProcessing:
    def __init__(self):
        self.sw = stopwords.words('indonesian')

    def remove_punctuation(self, text):
        '''a function for removing punctuation'''
        import string
        # replacing the punctuations with no space, 
        # which in effect deletes the punctuation marks 
        translator = str.maketrans('', '', string.punctuation)
        # return the text stripped of punctuation marks
        return text.translate(translator)

    def stemming(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        result = stemmer.stem(text)
        return result

    def remove_stopwords(self, word):
        '''a function for removing the stopword'''
        # removing the stop words and lowercasing the selected words
        if word.lower() not in self.sw:
            return word.lower()

    def freqWord(self, all_words):
        '''the input all_words should be in list '''
        result = nltk.FreqDist(all_words)
        return result

    def wordFeatures(self, all_words):
        result = list(all_words.keys())
        return result


class NLPModel:
    def __init__(self, training_set=None, testing_set=None):
        self.training_set = training_set
        self.testing_set = testing_set

    def originialNB_classifier(self):
        classifier = nltk.NaiveBayesClassifier.train(self.training_set)
        classifier.show_most_informative_features()

        return classifier

    def multinomialNB_classifier(self):
        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(self.training_set)

        return MNB_classifier

    def bernoulliNB_classifier(self):
        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(self.training_set)
 
        return BernoulliNB_classifier

    def logisiticRegression_classifier(self):
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(self.training_set)

        return LogisticRegression_classifier

    def SGD_classifier(self):
        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(self.training_set)

        return SGDClassifier_classifier

    def linearSVC_classifier(self):
        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(self.training_set)

        return LinearSVC_classifier

    def NuSVC_classifier(self):
        NuSVC_classifier = SklearnClassifier(NuSVC())
        NuSVC_classifier.train(self.training_set)
       
        return NuSVC_classifier
