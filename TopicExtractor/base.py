from abc import ABC, abstractmethod


class TopicExtractor(ABC):
    
    @abstractmethod
    def create_extractor():
        pass
    
    def fit_extractor():
        pass
    
    def fit_transform_extractor():
        pass
    
    def predict_extractor():
        pass