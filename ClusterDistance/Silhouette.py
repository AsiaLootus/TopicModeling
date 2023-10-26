from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from typing import List

class Silhouette():
    @staticmethod
    def calculate_distance(data:pd.DataFrame):
        sil_score=silhouette_score(np.array(data.embedding.to_list()),data.topic_id.to_list())
        return sil_score
    
    @staticmethod
    def get_best_clust_num(silhouette_scores:List[float]):
        return silhouette_scores.index(max(silhouette_scores))