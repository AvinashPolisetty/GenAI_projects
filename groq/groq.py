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
groq_api_keys="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"
astra_db_app_token= "AstraCS:rCKjqBeLoCSOUBgiPNMSrTyp:8b4d5ac5e078e595b141cbd3886fdd0d158421abc9a95c144d1b615171ad875b"
astra_db_id="a0d8a73e-6cc0-439d-a8c9-e511d8f2c3cc"
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


#groq and chat template
llm=ChatGroq(groq_api_key=groq_api_keys,
         model_name="mixtral-8x7b-32768")

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")



from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retriever=astra_vector_store.as_retriever()
document_chain=create_stuff_documents_chain(llm,prompt)
retrieval_chain=create_retrieval_chain(retriever,document_chain)

response=retrieval_chain.invoke("Explain the self-reflection")
print(response['answer'])
