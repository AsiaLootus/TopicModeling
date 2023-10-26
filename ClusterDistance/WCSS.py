from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List


class WCSS():
    
    # Funzione per calcolare il WCSS
    @staticmethod
    def calculate_distance(data:pd.DataFrame):
        similarities = WCSS.compute_cosine_similarity(data)
        
        # Calcola il WCSS come la somma dei quadrati delle dissimilarità
        wcss = np.sum(1 - similarities)
        
        return wcss
    
    @staticmethod
    def compute_cosine_similarity(data:pd.DataFrame):
        # Raggruppa il DataFrame per 'cluster_id' e calcola il centroide di ciascun cluster
        centroids = data.groupby('topic_id')['embedding'].apply(lambda x: np.mean(x.tolist(), axis=0))
        
        # Calcola la similarità del coseno tra ciascun embedding e il centroide del suo cluster
        similarities = data.apply(lambda row: cosine_similarity([row['embedding']], [centroids[row['topic_id']]])[0][0], axis=1)
        return similarities
    
    @staticmethod
    def get_best_clust_num(wcss_results:List[float]):
        # Trova il punto di "gomito" nel grafico del WCSS
        elbow_point = np.argmin(np.diff(wcss_results)) + 1  # Aggiungi 1 perché la differenza riduce la lunghezza di 1

        # Il numero ottimale di cluster è dato dall'indice del punto di "gomito"
        optimal_num_clusters = n_clusters_range[elbow_point - 1]
        return optimal_num_clusters