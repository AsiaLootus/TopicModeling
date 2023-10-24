import pandas as pd
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("my_logger")


import argparse
import json
from typing import Optional, Dict

from Preprocessor.base import Preprocessor
from TopicExtractor.BERTopic import BERTopicExtractor

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

def main(filename:str, method:str="bertopic"):
    
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
    
    logger.info("start topic extraction...")
    # initialize and fit the preprocessor
    topic_extractor = BERTopicExtractor()
    topic_extractor.fit_transform_extractor(list_text)
    embeddings = topic_extractor.compute_embeddings(list_text)
    
    logger.info("ended!")
    df_input = pd.DataFrame({"text":list_text, "topic_id": topic_extractor.topics, "probs": topic_extractor.probs, "embedding": embeddings.tolist()})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser.add_argument('--filename', required=False, default=r"datasets\json\1.json', help='Name of the json filename')
    parser.add_argument('--n_clust', required=False, default="auto", help='Number of clusters to be used')
    # parser.add_argument('--method', required=False, default="bertopic", help='Method of clustering to be used')

    args = parser.parse_args()

    main(args.filename)# args.method)
