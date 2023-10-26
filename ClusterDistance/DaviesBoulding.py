from sklearn.metrics import davies_bouldin_score
import numpy as np
import pandas as pd
from typing import List

class DaviesBoulding():
    @staticmethod
    def calculate_distance(data:pd.DataFrame):
        sil_score=davies_bouldin_score(np.array(data.embedding.to_list()),data.topic_id.to_list())
        return sil_score
    
    @staticmethod
    def get_best_clust_num(dav_score:List[float]):
        return dav_score.index(min(dav_score))
