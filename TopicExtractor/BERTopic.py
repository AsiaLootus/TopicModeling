import pandas as pd
import numpy as np

from bertopic import BERTopic
from bertopic.backend._utils import select_backend

from typing import Union, List
import logging

from .base import TopicExtractor


class BERTopicExtractor(TopicExtractor):
    def __init__(self, verbose:bool=True, language:str="italian", embedding_model:str="all-mpnet-base-v2", nr_topics:Union[str,int]='auto'):
        self.topic_model = None 
        self.verbose = verbose
        self.language = language
        self.embedding_model = embedding_model
        self.nr_topics = nr_topics
        self.fitted = False
        self.topics = None
        self.probs = None

    def fit_transform_extractor(self, list_text:List[str]):
        self.topic_model = BERTopic(verbose=self.verbose, language=self.language, embedding_model=self.embedding_model, nr_topics=self.nr_topics) 
        self.topics, self.probs = self.topic_model.fit_transform(list_text)
        self.fitted = True
    
    def fit_extractor(self, list_text:List[str]):
        self.topic_model.fit(list_text)
        self.fitted = True
    
    def transform_extractor(self, list_text:List[str]):
        if self.fitted:
            self.topics, self.probs = self.topic_model.transform(list_text)
        else:
            logging.info("Model not fitted, transform skipped")
        
    def compute_embeddings(self, list_text:List[str]) -> np.array:
        self.topic_model.embedding_model = select_backend(self.topic_model.embedding_model,
                                                language=self.topic_model.language)
        embeddings = self.topic_model._extract_embeddings(list_text,
                                                images=None,
                                                method="document",
                                                verbose=self.topic_model.verbose)

        # umap_embeddings = self.topic_model._reduce_dimensionality(embeddings, None)
        return embeddings
