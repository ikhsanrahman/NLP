'''import library'''
import pandas as pd 
import matplotlib.pyplot as plt
import nltk
import datetime
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

documents = []
all_words = []

featuresets  = []

#start process
start = datetime.datetime.now()

filename = 'datasets/hate_speech.txt'

#read dataset and label it based on column
df = pd.read_csv(filename, sep="\s+", names=['Lable','Tweet'])

# HS = df[df['Lable'] == 'HS'].shape[0]
# NON_HS = df[df['Lable'] == 'Non_HS'].shape[0]



# bar plot of the 2 classes
# plt.bar(HS,2, label="HS")
# plt.bar(NON_HS,2, label="NON_HS")
# plt.legend()
# plt.ylabel('Number of examples')
# plt.title('Propoertion of examples')
# plt.show()

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


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

# remove punctuation for each sentence
df['Tweet'] = df['Tweet'].apply(remove_punctuation)

sw = stopwords.words('indonesian')
def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

#remove stopword that exists in each sentence
df['Tweet'] = df['Tweet'].apply(stopwords)

#Label for each data. separated based on class 
HS = df[df['Lable'] == 'HS']
NON_HS = df[df['Lable'] == 'Non_HS']


# move each sentence that already labeled inside documents and all_words
for i in range(len(HS)):
    if i != None:
        result = HS.iloc[i]['Tweet']
        documents.append((result, 'HS'))

        token = word_tokenize(result)
        for w in token:
            all_words.append(w.lower())
for i in range(len(NON_HS)):
    if i != None:
        result = NON_HS.iloc[i]['Tweet']
        documents.append((result, 'NON_HS'))

        token = word_tokenize(result)
        for w in token:
            all_words.append(w.lower())

#save documents in pickle format that can be reused 
save_documents = open("pickle/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

#count how many words appear
all_words = nltk.FreqDist(all_words)
#get all keys of all_words and convert to list
word_features = list(all_words.keys())

#save word_features to pickle format that can be reused
save_word_features = open("pickle/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(documents):
    words = word_tokenize(documents)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# check the matching of each word
for rev, category in documents:
    feature = find_features(rev)
    featuresets.append((feature, category))

featuresets_f = open("pickle/featuresets.pickle", "wb")
pickle.dump(featuresets, featuresets_f)
featuresets_f.close()

# data example. divide data as training and testing      
training_set = featuresets[:500]
testing_set =  featuresets[200:]

# ------------------------------------------------   Modelling, Training, Testing Area  ----------------------------
# Training and testing using naive_bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# Save naive bayes to format pickle
save_classifier = open("pickle/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# Training and testing using multinomial naive_bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# Save multinomial Naive Bayes to format pickle
save_classifier = open("pickle/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# Training and testing using bernoulli Naive bayes
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# Save bernoulli Naive bayes to format pickle
save_classifier = open("pickle/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

# Training and testing using Logistic Regression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# Save Logistic Regression bayes to format pickle
save_classifier = open("pickle/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close() 

# Training and testing using stocastic gradient descent
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# Save stocastic gradient descent to format pickle
save_classifier = open("pickle/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()


# Training and testing using Linear SVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# Save linear SVC to format pickle
save_classifier = open("pickle/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# Training and testing using multinomial nu SVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


classifier = Classifier(
                        NuSVC_classifier,
                        LinearSVC_classifier,
                        MNB_classifier,
                        BernoulliNB_classifier,
                        LogisticRegression_classifier
                    )

print("classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)


end = datetime.datetime.now()
final_time = end-start

print("final_time", final_time)