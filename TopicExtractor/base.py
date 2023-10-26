from abc import ABC, abstractmethod


class TopicExtractor(ABC):
    
    @abstractmethod
    def fit_extractor():
        pass
    
    @abstractmethod
    def fit_transform_extractor():
        pass
    
    @abstractmethod
    def transform_extractor():
        pass
    
    @abstractmethod
    def compute_embeddings():
        pass