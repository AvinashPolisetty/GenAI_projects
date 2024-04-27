from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature":0.1,"max_length":10000}
)

question="what is spacex"
print(hf.invoke(question))