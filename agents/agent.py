from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models.google_palm import ChatGooglePalm
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"]= os.getenv("GOOGLE_API_KEY")


#default tools
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki_tool.name)


tools=[wiki_tool]

llm=ChatGooglePalm()

from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")









