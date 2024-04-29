import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Cassandra
import cassio
import bs4

load_dotenv()

#database initialization
groq_api_key=os.environ['GROQ_API_KEY']
astra_db_app_token= "AstraCS:FLAsxRtrLzbdyRMbRhayEMXT:a9bb94b186ff03b2e1c67f2368de26a372025f39bdf130002aeb772bc0608aeb"
astra_db_id="8e00d55c-6c90-473a-8f6d-b156e4f98d8f"
cassio.init(token=astra_db_app_token,database_id=astra_db_id)


#data ingestion
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content")
                     )))

text_docs=loader.load()
print(text_docs)


#text_splitting
text_split=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
docs=text_split.split_documents(text_docs)
docs[:3]


#embeddings

embeddings=OllamaEmbeddings(model="gemma:2b")
astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="q_a_application",
    session=None,
    keyspace=None
)


from langchain.indexes.vectorstore import VectorStoreIndexWrapper
astra_vector_store.add_documents(docs)
print("Inserted %i headlines." % len(docs))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)