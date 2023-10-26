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
from decouple import Config

from Preprocessor.base import Preprocessor
from TopicExtractor.BERTopic import BERTopicExtractor
from TopicExtractor.Top2Vec import Top2VecExtractor
from TopicExtractor.base import TopicExtractor
from ClusterDistance.WCSS import WCSS
from ClusterDistance.Silhouette import Silhouette
from ClusterDistance.DaviesBoulding import DaviesBoulding
from LabelAssignment.openai_assign import OpenAiAssign



def read_file(filename:str) -> Optional[Dict[str,str]]:
    # Opening JSON file
    data = None
    try:
        with open(filename, "r", encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                logger.error("Can't jsonify " + filename + ". Exception: "+str(e))
                  
    except Exception as e:
        logger.error("Can't read " + filename + ". Exception: "+str(e))
    
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

def define_topic_labels(df_input:pd.DataFrame, question:str="", n_top:int=100):
    labels = {}
    label_ids = list(df_input.topic_id.unique())
    chat = OpenAiAssign(question=question)
    for l in label_ids:
        if l != -1:
            df_small = df_input[(df_input.topic_id == l) & (df_input.probs == 1)]
            similarities = WCSS.compute_cosine_similarity(df_small)
            df_small["similarities"] = similarities
            df_small = df_small.sort_values("similarities", ascending=False)
            df_small = df_small.iloc[:n_top]
            
            label = chat.compute_question(df_small.text.to_list())
            labels[l] = label
    return labels

def read_question(filename_question:str):
    question = ""
    if filename_question != "":
        try:
            with open(filename_question, "r") as q:
                question = q.read()
        except Exception as e:
            logger.warning("Can't open question file")
    return question


def main(
        filename:str, 
        filename_question:str="", 
        n_clust:str="auto", 
        max_clust_num:int=26, 
        plot_results:bool=False, 
        method:str="bertopic",
        num_clust_method:str="wcss"):
    
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
    # elif method == "top2vec":
    #     topic_model = Top2VecExtractor
    
    if topic_model is None:
        logger.warning("Method choosen not available. Available methods: bertopic")
    
    # initialize model
    topic_extractor = None
    df_input = None
    
    if n_clust != 'compute':
        # compute using "auto" or predefined number of clusters
        if not n_clust == "auto":
            n_clust = int(n_clust) + 1
        topic_extractor, df_input = compute_topics(list_text, n_clust, topic_model=topic_model)
        
        optimal_num_clusters = len(df_input.topic_id.unique()) - 1 # tolgo il cluster -1
    
    else:
        # choose the best number of clusters
        if num_clust_method == "wcss":
            method_clust = WCSS
        elif num_clust_method == "silhouette":
            method_clust = Silhouette
        elif num_clust_method == "davies_boulding":
            method_clust = DaviesBoulding
        
        dist_results = []
        n_clusters_range = []
        for n_c in range(3, max_clust_num, 1):
            n_clusters_range.append(n_c)
            logger.info(f"compute {n_c} clusters...")
            topic_extractor, df_input = compute_topics(list_text, n_c, topic_model=topic_model)
            if -1 in list(df_input.topic_id.unique()):
                df_input = df_input[df_input.topic_id != -1]
            distance = method_clust.calculate_distance(df_input)
            dist_results.append(distance)
            
            # free space if cuda is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        optimal_num_clusters = method_clust.get_best_clust_num(dist_results)
        
        # topic_extractor = topic_extractors[elbow_point - 1]
        # df_input = df_inputs[elbow_point - 1]
        topic_extractor, df_input = compute_topics(list_text, optimal_num_clusters, topic_model=topic_model)

    logger.info(f"Numero ottimale di cluster: {optimal_num_clusters}")
        
    logger.info("Compute labels...")
    question = read_question(filename_question)
    labels = define_topic_labels(df_input, question)
    
    logger.info(f"topics: {labels}")
    
    if plot_results:
        # df_topic = topic_model.get_topic_info()
        fig3 = topic_extractor.topic_model.visualize_documents(list_text)
        fig3.show()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser.add_argument('--filename', required=False, default=r"data\datasets\json\1.json", help='Name of the json dataset filename')
    parser.add_argument('--filename_question', required=False, default=r"data\datasets\sq\1q.txt", help='Name of the question filename')
    parser.add_argument('--n_clust', required=False, default="compute", help='Number of clusters to be used')
    parser.add_argument('--num_clust_method', required=False, default="silhouette", help='Method used to compute the number of clusters to be used')
    # parser.add_argument('--method', required=False, default="bertopic", help='Method of clustering to be used')

    args = parser.parse_args()

    main(args.filename, args.filename_question, n_clust=args.n_clust, num_clust_method=args.num_clust_method)# args.method)