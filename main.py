import pandas as pd
import numpy as np
import torch
import gc

import logging
logging.basicConfig(
    level=logging.INFO,  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("my_logger")


import argparse
import json
from typing import Optional, Dict, Union, List

from Preprocessor.base import Preprocessor
from TopicExtractor.BERTopic import BERTopicExtractor
from TopicExtractor.base import TopicExtractor
from ClusterDistance.WCSS import WCSS




def read_file(filename:str) -> Optional[Dict[str,str]]:
    # Opening JSON file
    data = None
    try:
        with open(filename, "r", encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                logger.exception("Can't jsonify " + filename + ". Exception: "+str(e))
                  
    except Exception as e:
        logger.exception("Can't read " + filename + ". Exception: "+str(e))
    
    return data 

def compute_topics(list_text:List[str], nr_topics:Union[int,str]="auto", topic_model:TopicExtractor=BERTopicExtractor):
    logger.info("start topic extraction...")
    # initialize and fit the preprocessor
    topic_extractor = topic_model(nr_topics=nr_topics)
    topic_extractor.fit_transform_extractor(list_text)
    embeddings = topic_extractor.compute_embeddings(list_text)
    
    logger.info("ended!")
    df_input = pd.DataFrame({"text":list_text, "topic_id": topic_extractor.topics, "probs": topic_extractor.probs, "embedding": embeddings.tolist()})
    return topic_extractor, df_input


def main(filename:str, n_clust:str="auto", max_clust_num:int=26, plot_results:bool=False, method:str="bertopic"):
    
    # read file
    data = read_file(filename)
    if data is None:
        return 

    # convert to a list of string
    list_text = []
    if len(data) > 0 and "text" in data[0]:
        list_text = [l["text"] for l in data]
    else:
        logger.exception("Input json must contain a 'text' variable.")
        return
    
    logger.info("start preprocessing...")
    # preprocess the list
    list_text = Preprocessor.preprocess_data(list_text)
    
    # select model required
    topic_model = None
    if method == "bertopic":
        topic_model = BERTopicExtractor
    
    if topic_model is None:
        logger.exception("Method choosen not available. Available methods: bertopic")
    
    # initialize model
    topic_extractor = None
    df_input = None
    
    if n_clust != 'compute':
        # compute using "auto" or predefined number of clusters
        topic_extractor, df_input = compute_topics(list_text, n_clust, topic_model=topic_model)
    
    else:
        # choose the best number of clusters
        wcss_results = []
        n_clusters_range = []
        for n_c in range(2, max_clust_num, 2):
            n_clusters_range.append(n_c)
            logger.info(f"compute {n_c} clusters...")
            topic_extractor, df_input = compute_topics(list_text, n_c, topic_model=topic_model)
            wcss = WCSS.calculate_wcss(df_input)
            wcss_results.append(wcss)
            
            # clean space if cuda is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Trova il punto di "gomito" nel grafico del WCSS
        elbow_point = np.argmin(np.diff(wcss_results)) + 1  # Aggiungi 1 perché la differenza riduce la lunghezza di 1

        # Il numero ottimale di cluster è dato dall'indice del punto di "gomito"
        optimal_num_clusters = n_clusters_range[elbow_point - 1]
        
        # topic_extractor = topic_extractors[elbow_point - 1]
        # df_input = df_inputs[elbow_point - 1]
        topic_extractor, df_input = compute_topics(list_text, optimal_num_clusters, topic_model=topic_model)

        logger.info(f"Numero ottimale di cluster: {optimal_num_clusters}")
    
    if plot_results:
        # df_topic = topic_model.get_topic_info()
        fig3 = topic_extractor.topic_model.visualize_documents(list_text)
        fig3.show()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser.add_argument('--filename', required=False, default=r'data\datasets\json\1.json', help='Name of the json filename')
    parser.add_argument('--n_clust', required=False, default="compute", help='Number of clusters to be used')
    # parser.add_argument('--method', required=False, default="bertopic", help='Method of clustering to be used')

    args = parser.parse_args()

    main(filename=args.filename, n_clust=args.n_clust, plot_results=True)# args.method)