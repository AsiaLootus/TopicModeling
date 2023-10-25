from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class WCSS():
    
    # Funzione per calcolare il WCSS
    @staticmethod
    def calculate_wcss(data):
        similarities = WCSS.compute_cosine_similarity(data)
        
        # Calcola il WCSS come la somma dei quadrati delle dissimilarità
        wcss = np.sum(1 - similarities)
        
        return wcss
    
    @staticmethod
    def compute_cosine_similarity(data):
        # Raggruppa il DataFrame per 'cluster_id' e calcola il centroide di ciascun cluster
        centroids = data.groupby('topic_id')['embedding'].apply(lambda x: np.mean(x.tolist(), axis=0))
        
        # Calcola la similarità del coseno tra ciascun embedding e il centroide del suo cluster
        similarities = data.apply(lambda row: cosine_similarity([row['embedding']], [centroids[row['topic_id']]])[0][0], axis=1)
        return similarities