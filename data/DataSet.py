from __future__ import annotations
from typing import Dict, Any, List, Iterable, Union, Callable
from collections.abc import Iterable as IterableType
from data.Data import Data
import pandas as pd
import json
import re

class DataSet:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index: int) -> Data:
        if callable(index): return DataSet([ d for d in self.data if index(d) ])
        elif type(index) is slice: return DataSet(self.data[index])
        if index >= 0 and index < len(self.data) : return self.data[index]
        return Data.NONE

    def __iter__(self) -> Iterable[Data]: return iter(self.data)
    def __setitem__(self, index: int, value: Any) -> bool: return False
    def __len__(self) -> int: return len(self.data)
    def __str__(self) -> str: return 'DataSet<{}>'.format(len(self))
    def __repr__(self) -> str: return str(self)

    @staticmethod
    def CategoryFromScore(score: int) -> str:
        if score > 0: return 'positive'
        elif score < 0: return 'negative'
        return 'neutral'

    @staticmethod
    def FromJSON(path: str, keywordsPath: str, on_generate: Callable[Data, int, int]=None) -> DataSet:
        rawdata = []
        keywords = {}
        with open(keywordsPath, 'r', encoding='utf-8') as f: keywords = json.load(f)
        with open(path, 'r', encoding='utf-8') as f: rawdata = json.load(f)
        if type(rawdata) is not list: rawdata = []
        data = []
        index = 0
        raw_len = len(rawdata)
        for d in rawdata:
            if type(d) is not dict: continue
            source = d.get('source', '???').lower()
            title = d.get('title', '').lower()
            text = d.get('text', '').lower()
            score = 0
            for negative in keywords.get('negative', []):
                if text.find(negative.lower()) != -1: score -= 1
            
            for positive in keywords.get('positive', []):
                if text.find(positive.lower()) != -1: score += 1
            category = DataSet.CategoryFromScore(score)
            result = Data(text, category, source, title, score)
            if on_generate is not None: on_generate(result, index, raw_len)
            data.append(result)
            index += 1
        return DataSet(data)
    
    @staticmethod
    def FromDataFrame(dataframe: pd.DataFrame) -> DataSet:
        data = []
        dataview = dataframe.loc
        for index in dataframe.index:
            row = dataview[index]
            text = str.join(' ', [ i * row[i] for i in row.index if i != Data.CATEGORY_NAME and i != Data.SCORE_NAME ])
            if Data.CATEGORY_NAME in row: score = row[Data.CATEGORY_NAME]
            elif Data.SCORE_NAME in row: score = row[Data.SCORE_NAME]
            else: score = 0
            category = DataSet.CategoryFromScore(score)
            data.append(Data(text, category=category, score=score))
        return DataSet(data)
    
    @staticmethod
    def FromAny(text: Union[str, Iterable[str], Iterable[Data], pd.DataFrame], category: Union[str, Iterable[str]]=None) -> DataSet:
        data = None
        if type(text) is str and type(category) is str: data = DataSet([ Data(text, category) ])
        elif isinstance(text, pd.DataFrame): data = DataSet.FromDataFrame(text)
        elif isinstance(text, DataSet): data = text
        elif isinstance(text, IterableType):
            if type(category) is str: data = DataSet([ Data(x, category) for x in text ])
            elif isinstance(category, IterableType): data = DataSet([ Data(text[i], category[i]) for i in range(len(text)) ])
        if not data: raise Exception('The given input is not supported: <{}, {}>.\nUse <str, str>, <str[], str|str[]>, <pandas.Dataframe[words, word_count] | Data[], Unknown>'.format(type(text), type(category)))
        return data