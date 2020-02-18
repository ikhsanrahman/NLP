'''this module to train the dataset that has been processed to get clean data'''

import pickle
import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from nlp import PreProcessing, NLPModel, Classifier

start = datetime.datetime.now()

process = PreProcessing()

documents = []
all_words = []
featuresets = []


# open datasets
data_pos = open("datasets/id/positive.txt","r").read()
data_neg = open("datasets/id/negative.txt","r").read()

# stem each word inside datasets
data_pos = process.stemming(data_pos)
data_pos = process.stemming(data_neg)

# remove punctuation and collect to document and all_words
data_pos = process.remove_punctuation(data_pos)
data_neg = process.remove_punctuation(data_neg)

# remove stopwords
for data in data_pos.split():
    if data != None:
        stopword = process.remove_stopwords(data)
        if stopword != None:
            documents.append( (stopword, "pos") )

        token_pos = word_tokenize(data_pos)
        for w in token_pos:
            all_words.append(w.lower())

for data in data_neg.split():
    if data != None:
        stopword = process.remove_stopwords(data)
        if stopword != None:
            documents.append( (stopword, "neg") )

        token_neg = word_tokenize(data_neg)
        for w in token_neg:
            all_words.append(w.lower())

# save documents to pickle format
save_documents = open("pickle/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

# count each word inside all_words and list the keys
all_words = process.freqWord(all_words)
word_features = process.wordFeatures(all_words)

# save word to pickle format
save_word_features = open("pickle/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


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

model = NLPModel(training_set, testing_set)

NB = model.originialNB_classifier()
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB, testing_set))*100)

save_classifier = open("pickle/originalnaivebayes5k.pickle","wb")
pickle.dump(NB, save_classifier)
save_classifier.close()

MNB = model.multinomialNB_classifier()
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB, testing_set))*100)

save_classifier = open("pickle/MNB_classifier5k.pickle","wb")
pickle.dump(MNB, save_classifier)
save_classifier.close()

BNB = model.bernoulliNB_classifier()
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB, testing_set))*100)

save_classifier = open("pickle/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BNB, save_classifier)
save_classifier.close()

SGD = model.SGD_classifier()
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD, testing_set))*100)

save_classifier = open("pickle/SGDC_classifier5k.pickle","wb")
pickle.dump(SGD, save_classifier)
save_classifier.close()

LR = model.logisiticRegression_classifier()
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LR, testing_set))*100)

save_classifier = open("pickle/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LR, save_classifier)
save_classifier.close()

LSVC = model.linearSVC_classifier()
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LSVC, testing_set))*100)

save_classifier = open("pickle/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LSVC, save_classifier)
save_classifier.close()

classifier = Classifier(
                    NB,
                    LSVC,
                    MNB,
                    BNB,
                    LR
                )

print("classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

end = datetime.datetime.now()
final_time = end-start

print("final_time", final_time)