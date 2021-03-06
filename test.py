'''this module to train the dataset that has been processed to get clean data'''
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

start = datetime.datetime.now()

documents = []
all_words = []
featuresets = []

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

# open datasets
data_pos = open("datasets/id/positive.txt","r").read()
data_neg = open("datasets/id/negative.txt","r").read()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

data_pos = stemmer.stem(data_pos)
data_neg = stemmer.stem(data_neg)

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# remove punctuation and collect to document and all_words
data_pos = remove_punctuation(data_pos)
data_neg = remove_punctuation(data_neg)

sw = stopwords.words('indonesian')
def remove_stopwords(word):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    if word.lower() not in sw:
        return word.lower()


# remove stopwords
for data in data_pos.split():
    if data != None:
        stopword = remove_stopwords(data)
        if stopword != None:
            documents.append( (stopword, "pos") )

        token_pos = word_tokenize(data_pos)
        for w in token_pos:
            all_words.append(w.lower())

for data in data_neg.split():
    if data != None:
        stopword = remove_stopwords(data)
        if stopword != None:
            documents.append( (stopword, "neg") )

        token_neg = word_tokenize(data_neg)
        for w in token_neg:
            all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())

print(documents)
# print(all_words)

# find features 
def find_features(word):
    print(word)
    words = word_tokenize(word)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# check the matching of each word
for rev, category in documents:
    feature = find_features(rev)
    featuresets.append((feature, category))

# data example:      
training_set = featuresets[:4000]
testing_set =  featuresets[3800:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


end = datetime.datetime.now()
final_time = end-start

print("final_time", final_time)