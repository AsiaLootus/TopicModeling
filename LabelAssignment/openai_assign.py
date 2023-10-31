from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import openai
from decouple import Config
import logging
logger = logging.getLogger("TopicModeling")
# print("openai key:", openai.api_key)

class OpenAiAssign():
    
    def __init__(
                self, 
                llm_model:str="gpt-3.5-turbo", 
                question:str="", 
                template:str="""Giving the question: ```{question}```, give me a title that describes the answers. Give me the topic in the same language as the one in the texts in input.
                    Texts: ```{text}```
                    Topic: 
                    """,
                temperature:float=0.0
                ):
        # self.llm_model = llm_model
        config = Config('.env')
        openai.api_key = config.get('OPENAI_KEY')
        logger.info(config.get('OPENAI_KEY'))
        os.environ["OPENAI_API_KEY"] = config.get('OPENAI_KEY')
        self.question = question
        self.chat = ChatOpenAI(temperature=temperature, model=llm_model, request_timeout=30)
        self.prompt_template = ChatPromptTemplate.from_template(template)
        
    def _call_api(self, texts_concat:str):
        customer_messages = self.prompt_template.format_messages(
                                question=self.question,
                                text=texts_concat)
        customer_response = self.chat(customer_messages)
        return customer_response
    
    def compute_question(self, texts_concat:str):
        customer_response = None
        try:
            customer_response = self._call_api(texts_concat)
        except Exception as e:
            logger.exception(e)
        if customer_response != None:    
            tt = customer_response.content
            if tt[0] == '"':
                tt = tt[1:-2]
            return tt
        else:
            return ""
    
    