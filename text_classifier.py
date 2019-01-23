# -*- coding: utf-8 -*-
"""text_classifier.py: Classifies text comments regarding its polarity.
author: Willian Eduardo Becker
date: 09-08-2017
"""

import re
import sys
import time
import string
import argparse
from random import shuffle
from nltk.tokenize import WordPunctTokenizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

parser = argparse.ArgumentParser(
    description='SVM classifier for sentiment analysis of comments')
parser.add_argument('--classifier', dest='clf', metavar='c',
                    type=str, help='select svm or mnb or sgd classifier')
parser.add_argument('--neutral', dest='discard_neutrals', metavar='n',
                    type=bool, help='discard neutral comments', default=True)
args = parser.parse_args()

# CONSTANTS
FILE_PATH = "datasets/dataset_sample.txt"


def get_execution_time(spended_time):
    if spended_time > 60:
        minutes = int(spended_time/60)
        if minutes > 60:
            hours = int(minutes/60)
            print "It took " + str(hours) + " hours to execute."
        elif minutes == 1:
            print "It took 1 minute to execute."
        else:
            print "It took " + str(minutes) + " minutes to execute."
    elif int(spended_time) == 1:
        print "It took 1 second to execute."
    else:
        print "It took " + str(int(spended_time)) + " seconds to execute."


class Classifier:
    def __init__(self, clf=args.clf):
        self.stop_words = [unicode(x, "utf-8")[:-1]
                    for x in open("datasets/stop_words.txt", 'r').readlines()]
        self.clf = clf

    def get_data(self, path):
        # Read and organize data for test and training.
        text = open(path, 'r')
        rows = text.readlines()
        self.tuples = []
        pos_comments = []
        neu_comments = []
        neg_comments = []
        shuffle(rows)
        for row in rows:
            if row[-2] == '0':  # negative
                neg_comments.append((row[:-3], 0))
            elif row[-2] == '1':  # neutral
                neu_comments.append((row[:-3], 1))
            elif row[-2] == '2':  # positive
                pos_comments.append((row[:-3], 2))

        if args.discard_neutrals is True:
            print "Discarding neutral comments ..."
            min_len = min(len(pos_comments), len(neg_comments))
            self.tuples = pos_comments[:min_len] + neg_comments[:min_len]
        else:
            print "Considering neutral comments ..."
            min_len = min(len(pos_comments), len(neu_comments),
                          len(neg_comments))
            self.tuples = pos_comments[:min_len] \
                + neg_comments[:min_len] + neu_comments[:min_len]
        shuffle(self.tuples)

    def remove_stop_words(self, sent):
        # Remove stop words in sent.
        regex = re.compile(r'['+string.punctuation+'0-9]+')
        sent = re.sub(regex, " ", sent)
        sent = unicode(sent, "utf-8")
        sent = WordPunctTokenizer().tokenize(sent)
        for word in sent:
            if word in self.stop_words:
                sent.remove(word)
        return ' '.join(sent)

    def divide_data(self, test_size=0.2):
        # Divide data into train and test accordin to the variable test_size.
        X_train, X_test, y_train, y_test = \
            train_test_split([x for x, y in self.tuples],
                             [y for x, y in self.tuples],
                             test_size=test_size, random_state=42)
        self.train = zip(X_train, y_train)
        self.test = zip(X_test, y_test)

    def extract_features(self):
        # Extract features from text based on bag of words.
        # Get text from all comments.
        text = [self.remove_stop_words(x.lower()) for x, y in self.tuples]
        target = [y for x, y in self.tuples]
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(text)
        self.names = vectorizer.get_feature_names()
        self.tuples = zip(features.A, target)

    def train_data(self):
        # Train the classifier using the extracted features.
        if self.clf == "sgd" and args.discard_neutrals is True:
            self.classifier = SGDClassifier()
        elif self.clf == "mnb" and args.discard_neutrals is True:
            self.classifier = MultinomialNB()
        elif self.clf == "svm":
            self.classifier = svm.SVC()
        else:
            raise ValueError('Check out the parameters: classifier= %s, \
            discard_neutrals= %s', args.clf, args.discard_neutrals)
        x, y = [x for x, y in self.train], [y for x, y in self.train]
        self.classifier.fit(x, y)

    def evaluate(self):
        # Return the accuracy of the classifier.
        x, y_true = [x for x, y in self.test], [y for x, y in self.test]
        y_pred = []
        for sent in x:
            y_pred.append(self.classifier.predict(sent.reshape(1, -1)))

        print "Accuracy: " + str(accuracy_score(y_true, y_pred)) \
            + "\n============================"

    def classify(self, sent):
        # Predict the sentiment given a sentence.
        vec = CountVectorizer(vocabulary=self.names)
        sent = vec.fit_transform(sent)
        print self.classifier.predict(sent.A)

    def process(self, path):
        # Initialize the process of training and testing the svm classifier.
        start = time.time()
        self.get_data(path)
        print "Data extracted\n============================"
        self.extract_features()
        print "Features extracted\n============================"
        self.divide_data(0.1)
        print "Data divided\n============================"
        self.train_data()
        print "Trained data\n============================"
        self.evaluate()

        get_execution_time(time.time() - start)

if __name__ == "__main__":
    classifier = Classifier()
    classifier.process(FILE_PATH)

    #classify few unseen comments.
    classifier.classify(["Mt bom.".lower()])
    classifier.classify(["Horrível-- Prefiro o modo anterior.".lower()])
    classifier.classify(["Um dos melhores aplicativos já criados.".lower()])
    classifier.classify(["Aff--JÁ CANSEI DE ESPERAR!!!".lower()])
    classifier.classify(["Esse aplicativo é um lixo.".lower()])