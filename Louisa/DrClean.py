#!/usr/bin/env python
# coding: utf-8

# # TEXT data cleaner and tokenizer
# 
# we use NLTK, Regex, BeautifulSoup, inflect to clean the parsed text data (json file), tokenize them into sentences and restore them as JSON file.
# Resources used to create this code are as following:
# * <a href="https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html" target="_black" >  Text processing </a>
# *  <a href="https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/" target ="_black" >Tokenization </a>
# ---------------------------

# Step 1:Importing all the neccesary libraries

import pandas as pd
import re, string, unicodedata
import inflect
from collections import defaultdict
from bs4 import BeautifulSoup as BS
import spacy
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import json
import nltk.data
from nltk.tokenize import sent_tokenize 
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import numpy as np


# Step 2: Openning, premilinary cleaning, and tokenizing the text data into sentences


def open_clean_tokenize(filepath):
    """ filepath is the path to the json file to be openned. This function opens the file, do some basic cleaning (makes text lowercase, removes  and removes unwanted symboles and tokenize the data in to sentences
"""
    with open(filepath) as json_file:
        corpus = json.load(json_file)
        #corpus = pd.read_json(json_file, orient ='index') 
    corpus = str(corpus)
    corpus = BS(corpus, 'lxml').text #HTML decoding. BeautifulSoup's text attribute will return a string stripped of any HTML tags and metadata.
    corpus = corpus.lower()
    corpus = re.sub(r'\w*\d\w*', '', corpus)
    corpus = re.sub(r'http\S+', '', corpus) # removes url
    corpus = re.compile('[/(){}\[\]\|@;#:]').sub('', corpus) #replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    corpus = re.compile('(\$[a-zA-Z_][0-9a-zA-Z_]*)').sub('', corpus) #remove symbols from text.
    clean_tokenize = nltk.tokenize.sent_tokenize(corpus) # tokenizes data into sentences
    
    return clean_tokenize


# Step 3: Further cleaning (remove punctuations, numbers, and stop words) the tokenized data in setp 2 for to use in Word2vec


def remove_punctuation(corpus):
    """Remove punctuation from list of tokenized corpus"""
    new_corpus = []
    for token in corpus:
        new_token = re.sub(r'[^\w\s]', '', token)
        if new_token != '':
            new_corpus.append(new_token)
    return new_corpus

def replace_numbers(corpus):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_corpus = []
    for token in corpus:
        if token.isdigit():
            new_token = p.number_to_words(token)
            new_corpus.append(new_token)
        else:
            new_corpus.append(token)
    return new_corpus

def remove_stopwords(corpus):
    """Remove stop words from list of tokenized words"""
    new_corpus = []
    for token in corpus:
        if token not in stopwords.words('english'):
            new_corpus.append(token)
    return new_corpus

def normalize(corpus):
    corpus = remove_punctuation(corpus)
    corpus = replace_numbers(corpus)
    corpus = remove_stopwords(corpus)
    return corpus


# Step 4: Let's run the above two functions as below
# example: 
# 	  filepath = 'Basic Principles of Organic Chemistry_Roberts and Caserio'
# 	  corpus = open_clean_tokenize(filepath)
# 	  tokens = normalize(corpus)
#

# ## Stemming and lemmatization
# 
# We can further process the cleanned and tokenized data as follows:

def stem_words(tokens):
    """Stem words in list of tokenized words"""
    tokens = normalize(corpus)
    stemmer = LancasterStemmer()
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)
        stems.append(stem)
    return stems

def lemmatize_verbs(tokens):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, pos='v')
        lemmas.append(lemma)
    return lemmas


def stem_and_lemmatize(tokens):
    stems = stem_words(tokens)
    lemmas = lemmatize_verbs(tokens)
    return stems, lemmas


#example: final_data = stem_and_lemmatize(tokens)




