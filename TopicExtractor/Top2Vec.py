from top2vec import Top2Vec
from typing import Union, List
import logging
import pandas as pd
import numpy as np
import spacy

from .base import TopicExtractor

class Top2VecExtractor(TopicExtractor):
    def __init__(self, verbose:bool=True, language:str="italian", embedding_model:str="universal-sentence-encoder-multilingual", nr_topics:Union[str,int]='auto'):
        self.topic_model = None 
        self.verbose = verbose
        # self.language = language
        self.embedding_model = embedding_model
        self.nr_topics = nr_topics
        self.fitted = False
        self.topics = None
        self.probs = None
        self.embeddings = None
        
        self.stop_words = []
        if language == "italian":
            nlp = spacy.load('it_core_news_md')
            self.stop_words = spacy.lang.it.STOP_WORDS
        elif language == "english":
            nlp = spacy.load("en_core_web_sm")
            self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
            
        
        
    def fit_transform_extractor(self, list_text:List[str]):
        self.topic_model = Top2Vec(
            list_text,
            embedding_model=self.embedding_model,
            speed="deep-learn",
            # tokenizer=tok,
            ngram_vocab=True,
            ngram_vocab_args={"connector_words": list(self.stop_words)},
            verbose=self.verbose
        )
        self.embeddings = self.topic_model.document_vectors
        
    def fit_extractor(self, list_text:List[str]):
        raise Exception("Not implemented")
    
    def transform_extractor(self, list_text:List[str]):
        raise Exception("Not implemented")
    
    def compute_embeddings(self):
        return self.embeddings