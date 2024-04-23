from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.google_palm import ChatGooglePalm
from langchain_community.chat_models.openai import ChatOpenAI
from langserve import add_routes
import uvicorn

from dotenv import load_dotenv
import os
load_dotenv()


os.environ["GOOGLE_API_KEY"]= os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version=1.0,
    description="A simple API server"
)


add_routes(
    app,
    ChatOpenAI(),
    path="/googleplam"
)

model=ChatOpenAI()

prompt=ChatPromptTemplate.from_template("Write a essay about {topic} within 50 words")

add_routes(
    app,
    prompt|model,
    path="/essay"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
