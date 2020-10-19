from __future__ import annotations
from typing import Dict, List, Iterable, NewType, Union
from data.DataSet import Data, DataSet
from .TextClassifier import TextClassifier
import pandas as pd
import math
import json

class NaiveBayes(TextClassifier):
    '''Naive Bayes Network custom implementation, uses dictionary to store keys and probabilities'''

    def __add_category(self, category: str) -> NaiveBayes:
        '''setup the bayes network with a new category entry'''
        if category not in self.categories:
            self.documents[category] = 0
            self.word_count[category] = 0
            self.word_frequency[category] = {}
            self.categories[category] = True
            self.total_categories += 1
        return self
    
    def __add_token(self, token: str) -> NaiveBayes:
        if token not in self.vocabulary:
            self.vocabulary[token] = True
            self.total_words += 1
        return self

    def __dict_to_table(self, data: Dict[str, float], index='category') -> pd.DataFrame:
        '''Convert a dictionary in a table with columns as keys and values as rows'''
        columns = list(data)
        return pd.DataFrame([[ data[k] for k in columns ]], columns=columns, index=[index])

    def reset(self):
        '''Totally reset the Classifier'''
        self.vocabulary = {}
        self.categories = {}
        self.word_count = {}
        self.word_frequency = {}
        self.documents = {}
        self.total_documents = 0
        self.total_categories = 0
        self.total_words = 0
        return self

    def token_probability(self, token: str, category: str) -> float:
        '''return the probability of a given token to belong a given category'''

        # how many times this word has occurred in documents mapped to this category
        word_frequency = 0

        # what is the count of all words that have ever been mapped to this category
        word_count = 0

        if category in self.word_frequency:
            word_count = self.word_count[category]
            if token in self.word_frequency[category]:
                word_frequency = self.word_frequency[category][token]
        
        # use laplace Add-1 Smoothing equation
        # alpha = 1
        # d = word_count + 1 (prevents 0 division and takes in account category frequency)
        return (word_frequency + 1) / (self.total_words + word_count + 1)

    def category_probability(self, category: str) -> float:
        '''return the probability of the given category'''
        return self.documents[category] / self.total_documents if category in self.documents and self.total_documents != 0 else 0
    
    def word_probability(self, text: str) -> pd.DataFrame:
        '''retrive the probability table of the given text without knowing the probability of the category (no evidence): P(C | w1,...,wn)'''
        data = Data(text)
        probabilities = { k: 0 for k in self.categories }
        totalProbability = 0

        # iterate through our categories to find the one with max probability for this text
        for category in self.categories:
            # take the log to avoid underflow
            # product(x1, ..., xn) <= proportional => sum(log(x1), ..., log(xn))
            # product can underflow
            logProbability = 0

            # now determine P(w | c) for each word 'w' in the text
            for token, frequency in data.table.items():
                tokenProbability = self.token_probability(token, category)
                # determine the log of the P(w | c) for this word (no product but sum because of log)
                logProbability += frequency * math.log(tokenProbability)

            probability = logProbability
            probabilities[category] = probability
            totalProbability += probability
        
        return self.__dict_to_table(probabilities) / totalProbability if totalProbability != 0 else self.__dict_to_table({ k: 0 for k in self.categories })

    def probability(self, text: str) -> pd.DataFrame:
        '''retrive the probability table of the given text knowing the probability of categories: P(C) * P(C | w1,...,wn)'''
        data = Data(text)
        probabilities = { k: 0 for k in self.categories }
        totalProbability = 0

        # iterate through our categories to find the one with max probability for this text
        for category in self.categories:
            # start by calculating the overall probability of this category =>  out of all documents we've ever looked at, how many were mapped to this category
            categoryProbability = self.category_probability(category) or 1 / self.total_categories

            # take the log to avoid underflow
            # product(x1, ..., xn) <= proportional => sum(log(x1), ..., log(xn))
            # product can underflow
            logProbability = math.log(categoryProbability)

            # now determine P(w | c) for each word 'w' in the text
            for token, frequency in data.table.items():
                tokenProbability = self.token_probability(token, category)
                # determine the log of the P(w | c) for this word (no product but sum because of log)
                logProbability += frequency * math.log(tokenProbability)

            probability = math.exp(logProbability)
            probabilities[category] = probability
            totalProbability += probability
        
        return self.__dict_to_table(probabilities) / totalProbability if totalProbability != 0 else self.__dict_to_table({ k: 0 for k in self.categories })

    def fit(self, text: Union[str, Iterable[str], Iterable[Data], pd.DataFrame], category: Union[str, Iterable[str]]=None) -> NaiveBayes:
        '''learn probabilities for tokens extracted by the given text'''
        data = DataSet.FromAny(text, category)
        for d in data:
            # ensure we have defined the c category
            self.__add_category(d.category)
            # update our count of how many documents mapped to this category
            self.documents[d.category] += 1
            # update the total number of documents we have learned from
            self.total_documents += 1

            # Update our vocabulary and our word frequency count for this category
            for token, frequency in d.table.items():
                # add this word to our vocabulary if not already existing
                self.__add_token(token)

                # update the frequency information for this word in this category
                if token not in self.word_frequency[d.category]: self.word_frequency[d.category][token] = frequency
                else: self.word_frequency[d.category][token] += frequency
            
                # update the count of all words we have seen mapped to this category
                self.word_count[d.category] += frequency

        return self

    def words(self, categories: Union[str, Iterable[str]]) -> pd.DataFrame:
        '''return a sorted by probability table with tokens as columns and probability as rows for the given categories'''
        if type(categories) is str: categories = [categories]
        probabilities = [[0 for _ in categories] for _ in self.vocabulary ]
        for j, category in enumerate(categories):
            if category not in self.categories: continue
            category_probability = self.documents[category] / self.total_documents
            for i, token in enumerate(self.vocabulary):
                current_prob = category_probability * self.token_probability(token, category)
                probabilities[i][j] = current_prob

        return pd.DataFrame(probabilities, columns=categories, index=list(self.vocabulary))

    def from_json(self, data: Dict) -> NaiveBayes:
        self.vocabulary = data.get('vocabulary', self.vocabulary)
        self.categories = data.get('categories', self.categories)
        self.word_count = data.get('word_count', self.word_count)
        self.word_frequency = data.get('word_frequency', self.word_frequency)
        self.documents = data.get('documents', self.documents)
        self.total_documents = data.get('total_documents', self.total_documents)
        self.total_categories = data.get('total_categories', self.total_categories)
        self.total_words = data.get('total_words', self.total_words)
        return self

    def __str__(self) -> str:
        return 'NaiveBayes<{}, {}>[{}]'.format(self.total_documents, self.total_words, str.join(', ', self.categories))
    
    def __repr__(self) -> str:
        return str(self)