#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Created on 04/19/2019

@author: Datan Xu
'''
import re
import nltk
import os
from nltk.corpus import sentence_polarity
import random
import pandas as pd
from nltk.metrics import *

# Data Pre-processing

# open and read baby.txt
filepath='./baby.txt'
babyFile = open(filepath, 'r')
babyText = babyFile.read()
babyFile.close()

pattern = re.compile('reviewText:(.*)\n.*\n.*\n.*\nreviewTime.*, 2008')
reviews_2008 =re.findall(pattern, babyText)
# number of reviews
print("The number of Amazon Product Reviews in the year of 2008: ", len(reviews_2008))

# tokenize the reviews by nltk.sent_tokenize()
reviewSentences = []
for item in reviews_2008:
    reviewSentences += nltk.sent_tokenize(item)
print("The number of sentences: ", len(reviewSentences))

# tokenize each sentence by nltk.word_tokenize()
reviewWords = []
for item in reviewSentences:
    reviewWords.append(nltk.word_tokenize(item))

# Sentiment Analysis

# Case 1. Unigram features

# - We start by loading the sentence_polarity corpus and creating 
# - a list of documents where each document represents a single 
# - sentence with the words and its label.
sentences = sentence_polarity.sents()
sentence_polarity.categories()

# - The movie review sentences are not labeled individually, but 
# - can be retrieved by category. We first create the list of 
# - documents where each document(sentence) is paired with its label.
documents = [(sent, cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]

# - Since the documents are in order by label, we mix them up for 
# - later separation into training and test sets.
random.shuffle(documents)

# - We need to define the set of words that will be used for features.
# - This is essentially all the words in the entire document collection, 
# - except that we will limit it to the 2000 most frequent words. Note 
# - that we lowercase the words, but do not do stemming or remove stopwords.
# - The data set will be filtered by stopwords in the subsequent experiments.
all_words_list = [word for (sent, cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]

# - Now we can define the features for each document, using just the words, 
# - sometimes called the BOW or unigram features. The feature label will be 
# - ‘contains(keyword)’ for each keyword (aka word) in the word_features set, 
# - and the value of the feature will be Boolean, according to whether the 
# - word is contained in that document.
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# - Define the feature sets for the documents
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

# - We create the training and test sets, train a Naïve Bayes classifier, and 
# - look at the accuracy, and this time we’ll do a 90/10 split of our approximately 10,000 documents.
train_set, test_set = featuresets[1000:], featuresets[:1000]
unigramClassifier = nltk.NaiveBayesClassifier.train(train_set)

print("\nThe accuracy of Unigram Features: ", nltk.classify.accuracy(unigramClassifier, test_set))


# Case2. Subjectivity count features
# - We will first read in the subjectivity words from the subjectivity lexicon 
# - file created by Janyce Wiebe and her group at the University of Pittsburgh in the MPQA project.
# - Create a path variable to where you stored the subjectivity lexicon file.
SLpath = 'subjclueslen1-HLTEMNLP05.tff'

# - Now define and run the readSubjectivity() function that reads the file. It creates a 
# - Subjectivity Lexicon that is represented here as a dictionary, where each 
# - word is mapped to a list containing the strength, POStag, whether it is stemmed and the polarity.

# The data structure that is created is a dictionary where
#    each word is mapped to a list of 4 things:  
#        strength, which will be either 'strongsubj' or 'weaksubj'
#        posTag, either 'adj', 'verb', 'noun', 'adverb', 'anypos'
#        isStemmed, either true or false
#        polarity, either 'positive', 'negative', or 'neutral'

def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

SL = readSubjectivity(SLpath)

# - Now we create a feature extraction function that has all the word features
# - as before, but also has two features ‘positivecount’ and ‘negativecount’. 
# - These features contains counts of all the positive and negative subjectivity 
# - words, where each weakly subjective word is counted once and each strongly subjective word is counted twice.
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features

# - Now we create feature sets as before, but using this feature extraction function.
SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in documents]
train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
SL_classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(SL_classifier, test_set)
print("The accuracy of Subjectivity Count Features: ", nltk.classify.accuracy(SL_classifier, test_set))

# Case3. Negation features
# My strategy with negation words is to negate the word following the negation word, and
# we go through the document words in order adding the word features, but if the word 
# follows a negation words, change the feature to negated word.

# The form of some of the words is a verb followed by n’t. Now in the Movie Review Corpus 
# itself, the tokenization has these words all split into 3 words, e.g. “couldn”, “’”, and “t”. 
# (and I have a NOT_features definition for this case). But in this sentence_polarity corpus, 
# the tokenization keeps these forms of negation as one word ending in “n’t”.
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

# - Start the feature set with all 2000 word features and 2000 Not word features set to false. 
# - If a negation occurs, add the following word as a Not word feature (if it’s in the top 2000 feature words), 
# - and otherwise add it as a regular feature word.
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features

# - Then create feature sets as before, using the NOT_features extraction funtion, train the classifier and test the accuracy.
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
NOT_classifier = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Negation Features: ", nltk.classify.accuracy(NOT_classifier, test_set))


# Case 4. Filter by Stop Words (Based on Negation Features)

# Dataset is filtered by stop words which are not in the "negationwords" list
stopwords = nltk.corpus.stopwords.words('english')
newStopwords = [word for word in stopwords if word not in negationwords]
filtered_all_words_list = [word for word in all_words_list if word not in newStopwords]

# Select most 2000 frequent words to be the new word features
filtered_all_words = nltk.FreqDist(filtered_all_words_list)
filtered_word_items = filtered_all_words.most_common(2000)
filtered_word_features = [word for (word, count) in filtered_word_items]

# Redefine the NOT_features() function because of the change of the word features
def New_NOT_features(document, new_word_features, negationwords):
    features = {}
    for word in filtered_word_features:  # 2000 most frequent words
        features['contains({})'.format(word)] = False  # contains({word})=false
        features['contains(NOT{})'.format(word)] = False  # contains(NOT{word})=false
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        # make sure there is at least one word
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1  # get the word after negation word
            features['contains(NOT{})'.format(document[i])] = (document[i] in filtered_word_features)
        else:
            features['contains({})'.format(word)] = (word in filtered_word_features)
    return features

# - Then create feature sets as before, using the New_NOT_features extraction function, train the classifier and test the new accuracy.
Modified_NOT_featuresets = [(New_NOT_features(d, filtered_word_features, negationwords), c) for (d, c) in documents]
train_set, test_set = Modified_NOT_featuresets[1000:], Modified_NOT_featuresets[:1000]
Modified_NOT_classifier = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Negation Features filtered by stop words: ", nltk.classify.accuracy(Modified_NOT_classifier, test_set))


# Apply modified negation features to perform Sentiment Analysis of Amazon Product Reviews
pos = []
neg = []
for review in reviewWords:
    if (Modified_NOT_classifier.classify(New_NOT_features(review,filtered_word_features, negationwords)) == 'pos'):
        # use string.join() to combine list
        pos.append(" ".join(review))
    if (Modified_NOT_classifier.classify(New_NOT_features(review, filtered_word_features, negationwords)) == 'neg'):
        neg.append(" ".join(review))

print("\nThe number of positive sentences: ", len(pos))
print("The number of negative sentences: ", len(neg))

# Create two CSV files to store the positive and negative lists of sentences
posDataFrame = pd.DataFrame({'$Positive': pos})
negDataFrame = pd.DataFrame({'$Negative': neg})
posDataFrame.to_csv('PostiveReviews.csv', index=True)
negDataFrame.to_csv('NegativeReviews.csv', index=True)


# Use cross-validation to obtain precision, recall, and F-measure scores

# - First we build the reference and test lists from the classifier on the test set
reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(Modified_NOT_classifier.classify(features))

# - Now we use the NLTK function to define the confusion matrix, and we print it out
print("\nThe confusion matrix: ")    
cm = ConfusionMatrix(reflist, testlist)
print(cm)
# - Here we set up reference and test sets for each label that use the index number as the item identifiers.
refpos = set([i for i,label in enumerate(reflist) if label == 'pos'])
refneg = set([i for i,label in enumerate(reflist) if label == 'neg'])
testpos = set([i for i,label in enumerate(testlist) if label == 'pos'])
testneg = set([i for i,label in enumerate(testlist) if label == 'neg'])

# - Now to get precision, recall and F-measure for one of the labels, we must give the
# - reference and test sets for that label. It is easiest to define a function that calls the
# - three NLTK functions.
def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset))
    print(label, 'F-measure:', f_measure(refset, testset))
    
print("\nPrecision, recall, and F-measure scores for the two labels: ")   
printmeasures('pos', refpos, testpos)
printmeasures('neg', refpos, testpos)
