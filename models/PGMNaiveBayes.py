from __future__ import annotations
from typing import Dict, List, Iterable, NewType, Optional, Union
from pgmpy.models import NaiveBayes
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from data.DataSet import Data, DataSet
from .TextClassifier import TextClassifier
import pandas as pd
import numpy as np

class PGMNaiveBayes(TextClassifier):

    def __add_category(self, categories: Union[str, List[str], List[(str, str)], Dict[str, str]]) -> PGMNaiveBayes:
        '''setup the bayes network with a new category entry'''
        if type(categories) is str: categories = [categories]
        if type(categories) is dict: categories = categories.items()
        to_create = False
        for category in categories:
            if type(category) is str: category = (category, category)
            category, index = category
            if category not in self.categories:
                self.categories[category] = index
                self.cardinality = len(self.categories) or 1
                to_create = True
        if to_create: self.__create_class_cpd()
        return self
    
    def __add_token(self, tokens: Union[str, List[str]]) -> PGMNaiveBayes:
        if type(tokens) is str: tokens = [tokens]
        to_create = []
        for token in tokens:
            if token not in self.tokens:
                to_create.append(token)
                self.total_tokens += 1
        self.__create_word_cpd(to_create)
        return self
    
    def __create_word_cpd(self, tokens: Union[str, List[str]], check: bool=True) -> PGMNaiveBayes:
        '''Generate the table for the given token node'''
        if type(tokens) is str: tokens = [tokens]
        cpds = []
        for token in tokens:
            if token in self.tokens:
                self.model.remove_cpds(self.tokens[token])
            cpd_word = TabularCPD(
                variable=token,
                variable_card=2,
                evidence=[Data.CATEGORY_NAME],
                evidence_card=[self.cardinality],
                values=[ [0.5 for _ in range(self.cardinality) ] ] * 2
            )
            self.tokens[token] = cpd_word
            cpds.append(cpd_word)
        
        self.model.add_nodes_from(tokens)
        self.model.add_edges_from([ (Data.CATEGORY_NAME, token) for token in tokens ])
        self.model.add_cpds(*cpds)
        # if check: self.model.check_model()
        return self

    def __create_class_cpd(self, check: bool=True) -> PGMNaiveBayes:
        '''Generate the table for the category node'''
        if self.cpd_class:
            self.model.remove_cpds(self.cpd_class)
        self.cpd_class = TabularCPD(
            variable=Data.CATEGORY_NAME,
            variable_card=self.cardinality,
            values=[ [1 / self.cardinality] for _ in range(self.cardinality) ]
        )
        self.model.add_cpds(self.cpd_class)
        # if check: self.model.check_model()
        return self

    def __cpd_to_json(self, cpd: TabularCPD) -> Dict:
        return {
            'variable': cpd.variable,
            'variables': cpd.variables,
            'variable_card': cpd.variable_card.tolist(),
            'values': cpd.values.tolist()
        }
    
    def __cpd_from_json(self, cpd: Dict) -> TabularCPD:
        return TabularCPD(**cpd)

    def reset(self) -> PGMNaiveBayes:
        '''Totally reset the Classifier'''
        self.categories = {}
        self.tokens = {}
        self.cardinality = 1
        self.total_documents = 0
        self.total_tokens = 0
        self.cpd_class = None
        self.model = NaiveBayes()
        self.model.add_node(Data.CATEGORY_NAME)
        return self

    def token_probability(self, token: str, category: str) -> float:
        '''return the probability of a given token to belong a given category'''
        probability = self.model.predict_probability(pd.DataFrame([[1]], columns=[token]))
        column = '{}_{}'.format(Data.CATEGORY_NAME, self.categories.get(category, 0))
        return probability[column][0] if column in probability else 0

    def category_probability(self, category: str) -> float:
        '''return the probability of the given category'''
        score = Data.CATEGORY_VALUES.get(category, 0)
        elimination = VariableElimination(self.model)
        probability = elimination.query(variables=[Data.CATEGORY_NAME])
        state = probability.get_state_no(Data.CATEGORY_NAME, self.categories.get(category, 0))
        return probability.values[state]
    
    def word_probability(self, text: str) -> pd.DataFrame:
        '''retrive the probability table of the given text without knowing the probability of the category (no evidence): P(C | w1,...,wn)'''
        data = Data(text)
        elimination = VariableElimination(self.model)
        values = [ [] for _ in range(self.cardinality) ]
        for token in data.tokens:
            if token not in self.tokens:
                for v in values: v.append(1 / (self.cardinality or 1))
            else:
                probability = elimination.query(variables=[Data.CATEGORY_NAME], evidence={ token: 1 }).values
                for i in range(len(probability)): values[i].append(probability[i])

        return pd.DataFrame(np.array(values).T, columns=list(self.categories), index=data.tokens)
    
    def probability(self, text: str) -> pd.DataFrame:
        '''retrive the probability table of the given text knowing the probability of categories: P(C) * P(C | w1,...,wn)'''
        data = Data(text)
        values = pd.DataFrame([[ 1 if t in data.table else 0 for t in self.tokens ]], columns=self.tokens)
        probabilities = self.model.predict_probability(values)
        return probabilities.rename(columns={ '{}_{}'.format(Data.CATEGORY_NAME, v): k for k, v in self.categories.items() })
    
    def fit(self, text: Union[str, Iterable[str], Iterable[Data], pd.DataFrame], category: Union[str, Iterable[str]]=None) -> TextClassifier:
        '''learn probabilities for tokens extracted by the given text'''
        data = DataSet.FromAny(text, category)
        
        categories = []
        tokens = {}
        values = []
        
        for d in data:
            categories.append((d.category, d.score))
            for token in d.tokens: tokens[token] = 1
            values.append((d.table, d.score))
            self.total_documents += 1
        
        tokens = list(tokens)
        self.__add_category(categories)
        self.__add_token(tokens)
        
        data_values = [ [1 if t in v[0] else 0 for t in tokens ] + [v[1]] for v in values ]

        tokens.append(Data.CATEGORY_NAME)

        data_values = pd.DataFrame(data_values, columns=tokens)
        
        self.model.fit(data_values, Data.CATEGORY_NAME)

        return self

    def words(self, categories: Union[str, Iterable[str]]) -> pd.DataFrame:
        '''return a sorted by probability table with tokens as rows and categories as columns, for the given categories'''
        elimination = VariableElimination(self.model)
        values = [ [] for _ in range(self.cardinality) ]
        for token in self.tokens:
            probability = elimination.query(variables=[Data.CATEGORY_NAME], evidence={ token: 1 }).values
            for i in range(len(probability)): values[i].append(probability[i])

        return pd.DataFrame(np.array(values).T, columns=list(self.categories), index=list(self.tokens))

    def to_json(self) -> Dict:
        return {
            'categories': self.categories,
            'total_documents': self.total_documents,
            'tokens': { c.variable: c.values.tolist() for c in self.model.get_cpds() if c.variable != Data.CATEGORY_NAME },
        }
    
    def from_json(self, data: Dict) -> PGMNaiveBayes:
        self.total_documents = data.get('total_documents', self.total_documents)
        self.__add_category(data.get('categories', {}))
        self.model.remove_cpds(self.cpd_class)
        self.cpd_class = TabularCPD(**data.get('class')) if 'class' in data else self.cpd_class
        self.model.add_cpds(self.cpd_class)
        tokens = data.get('tokens', {})
        self.__add_token(list(tokens))
        cpds = { c.variable: c for c in self.model.get_cpds() }
        for token, values in tokens.items():
            if token in cpds:
                cpds[token].values = np.array(values)[0:self.cardinality,0:self.cardinality]

        self.model.check_model()
        return self

    def __str__(self) -> str:
        return 'NaiveBayes<{}, {}>[{}]'.format(self.total_documents, self.total_tokens, str.join(', ', self.categories))

    def __repr__(self) -> str:
        return str(self)