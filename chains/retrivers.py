from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()

from langchain.chains import create_retrieval_chain

loader = PyPDFLoader('transformer.pdf')
docs = loader.load()
print(docs)

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
documents=text_splitter.split_documents(docs)
print(documents)


db=Chroma.from_documents(documents[:30],HuggingFaceEmbeddings())

#llm models
llm=Ollama(model='gemma:2b')

# prompt
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")



from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt) 


#retriver

retriever=db.as_retriever()
retriever



retrieval_chain=create_retrieval_chain(retriever,document_chain)
response=retrieval_chain.invoke({"input":"Model Architecture"})
print(response['answer'])


