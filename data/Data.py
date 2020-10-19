from __future__ import annotations
from typing import Dict, Any, List, Iterable, Union
import re

Sanitizers = [
    re.compile(r'[^a-zA-Z\s]'),
    re.compile(r'https?\S*'),
    re.compile(r'[\(\)\@\#\d\:\'\"\?\!\.\,\-\_\|\\\/\^\$\&\£\€\[\]\{\}\«\»\%\…\“]')
]
Tokenizer = re.compile(r'[\s]+')

# remove punctuation from text (anything that isn't a word char or a space), tokeninze it by splitting every space
def tokenize(text: str) -> List[str]:
    for sanitizer in Sanitizers: text = sanitizer.sub('', text)
    return [ x for x in (t.replace(' ', '') for t in Tokenizer.split(text)) if x ]

# return the frequency table of the given list of tokens
def frequencyTable(tokens: Union[Iterable[str], str]) -> Dict[str, int]:
    if type(tokens) is str: tokens = tokenize(tokens)
    frequencyTable = {}
    for token in tokens:
        if token not in frequencyTable: frequencyTable[token] = 1
        else: frequencyTable[token] += 1
    return frequencyTable

class Data:
    CATEGORY_NAME = '__CATEGORY__'
    SCORE_NAME = '__SCORE__'
    CATEGORY_VALUES = {
        'positive': 1, 1: 'positive',
        'negative': -1, -1: 'negative',
        'neutral': 0, 0: 'neutral'
    }

    def __init__(self, text: Union[str, Data], category: str='???', source: str='__local__', title: str='', score: int=None):
        if isinstance(text, Data):
            self.source = text.source
            self.title = text.title
            self.text = text.text
            self.score = text.score
            self.category = text.category
            self.tokens = list(text.tokens)
            self.table = dict(text.table)
        else:
            self.source = source
            self.title = title
            self.text = text
            self.category = category
            self.score = score if score is not None else Data.CATEGORY_VALUES.get(self.category, 0)
            self.tokens = tokenize(self.text)
            self.table = frequencyTable(self.tokens)

    def __str__(self) -> str: return 'Data<{}, {}, {}>: {} - {}'.format(self.source, self.category, self.score, self.title[:10] + '...' if len(self.title) > 10 else '', self.text[:10] + '...' if len(self.text) > 10 else '')
    def __repr__(self) -> str: return str(self)

Data.EMPTY = Data('')