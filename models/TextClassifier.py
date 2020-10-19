from __future__ import annotations
from typing import Union, Dict, Iterable
from pandas import DataFrame
from data.Data import Data
from abc import ABC, abstractmethod
import json

class TextClassifier(ABC):
    '''Abstract class of a Text Classifier'''

    def __init__(self, path: str=None):
        self.reset()
        if path is not None: self.load(path)

    def reset(self) -> TextClassifier:
        '''Totally reset the Classifier'''
        return self

    @abstractmethod
    def token_probability(self, token: str, category: str) -> float:
        '''return the probability of a given token to belong a given category'''
        raise Exception('{}.token_probability(self, token: str, category: str) -> float IS NOT IMPLEMENTED'.format(self.__class__))

    @abstractmethod
    def category_probability(self, category: str) -> float:
        '''return the probability of the given category'''
        raise Exception('{}.category_probability(self, category: str) -> float IS NOT IMPLEMENTED'.format(self.__class__))
    
    @abstractmethod
    def word_probability(self, text: str) -> DataFrame:
        '''retrive the probability table of the given text without knowing the probability of the category (no evidence): P(C | w1,...,wn)'''
        raise Exception('{}.word_probability(self, text: str) -> pandas.DataFrame IS NOT IMPLEMENTED'.format(self.__class__))
    
    @abstractmethod
    def probability(self, text: str) -> DataFrame:
        '''retrive the probability table of the given text knowing the probability of categories: P(C) * P(C | w1,...,wn)'''
        raise Exception('{}.probability(self, text: str) -> pandas.DataFrame IS NOT IMPLEMENTED'.format(self.__class__))
    
    def predict(self, text: str, categoryEvidence=False) -> (str, float):
        '''
        retrive the most probable category of the given text, using the unconditioned probability ( P(C|w1,...,wn) )\n
        If categoryEvidence is set to True, it takes in account the current probability to belong to a category ( P(C) * P(C|w1,...,wn))
        '''
        probabilities = self.probability(text) if categoryEvidence else self.word_probability(text)
        column = probabilities.idxmax(axis=1)[0]
        return column, probabilities[column]['category']

    @abstractmethod
    def fit(self, text: Union[str, Iterable[str], Iterable[Data], DataFrame], category: Union[str, Iterable[str]]=None) -> TextClassifier:
        '''learn probabilities for tokens extracted by the given text'''
        raise Exception('{}.fit(self, text: str | str[] | Data[] | pandas.DataFrame, category?: str | str | None) -> self IS NOT IMPLEMENTED'.format(self.__class__))

    @abstractmethod
    def words(self, categories: Union[str, Iterable[str]]) -> DataFrame:
        '''return a sorted by probability table with tokens as rows and categories as columns, for the given categories'''
        raise Exception('{}.words(self, categories: str | str[]) -> pandas.DataFrame IS NOT IMPLEMENTED'.format(self.__class__))

    @abstractmethod
    def from_json(self, data: Dict) -> TextClassifier:
        '''Get information from a JSON readable format object to build the network'''
        raise Exception('{}.fromJSON(self, data: Dict) -> self IS NOT IMPLEMENTED'.format(self.__class__))

    def to_json(self) -> Dict:
        '''Convert the object to a JSON writtable format'''
        return self.__dict__

    def save(self, path: str) -> TextClassifier:
        '''save the network table in the target JSON file'''
        with open(path, 'w', encoding='utf-8') as file: json.dump(self.to_json(), file)
        return self

    def load(self, path: str) -> TextClassifier:
        '''load the network table from JSON the target file'''
        self.reset()
        data = None
        with open(path, 'r', encoding='utf-8') as file: data = json.load(file)
        if data is not None and type(data) == dict: self.from_json(data)
        return self