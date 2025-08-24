import os
from typing import List
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging

from models.Base import BaseModel

# 创建日志记录器
logger = logging.getLogger(__name__)

class KimiK2Model(BaseModel):
    def __init__(self, 
                 model_id="Kimi-K2-Instruct", 
                 model_api_version='2024-06-01', 
                 api_key=None):
        assert api_key is not None, "no api key is provided."
        self.model_id = model_id
        self.model_api_version = model_api_version

        headers = {
            #'Authorization': 'wisemodel-xxvqzbsnecjtoxufxodx',
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }        

        self.client = openai.OpenAI(
            #api_key = "wisemodel-xxvqzbsnecjtoxufxodx",
            api_key=api_key,
            base_url = "https://laiyeapi.aifoundrys.com:7443/v1",
            # base_url = "https://api.moonshot.cn/v1",
            default_headers = headers
        )
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, 
                 messages: List, 
                 temperature=0, 
                 presence_penalty=0, 
                 frequency_penalty=0, 
                 max_tokens=5000) -> str:
        logger.info(f"Sending request to model {self.model_id} with {len(messages)} messages")
        logger.debug(f"Messages content: {messages}")
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            stream=False,
            max_tokens=max_tokens,
        )
        
        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            error_msg = "No response choices returned from the API."
            logger.error(error_msg)
            raise ValueError(error_msg)

        result = response.choices[0].message.content
        logger.info(f"Received response from model {self.model_id}, response length: {len(result)} characters")
        logger.debug(f"Response content: {result[:200]}..." if len(result) > 200 else f"Response content: {result}")
        
        return result