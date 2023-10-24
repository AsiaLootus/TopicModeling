import pandas as pd
from typing import List
from langdetect import detect

class Preprocessor():
    
    @staticmethod
    def preprocess_data(list_text:List[str], lang:str="it") -> List[str]:
        list_text_ret = []
        for s in list_text:
            lang = Preprocessor._detect_lang(str(s))
            if lang == "it":
                list_text_ret.append(s)
        return list_text_ret
    
    @staticmethod
    def _detect_lang(s):
        return detect(str(s))