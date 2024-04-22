from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")



prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant,please response the given queries"),
        ("user","Question:{question}")
    ]
)


st.title("Chatbot with OpenAI LLM")
input_text=st.text_input("Ask the questions you have")


llm=ChatOpenAI(model="gpt-3.5-turbo")
output=StrOutputParser()
chain=prompt|llm|output

if input_text:
    st.write(chain.invoke({'question':input_text}))