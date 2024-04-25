from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


loader = PyPDFLoader('transformer.pdf')
docs = loader.load()
print(docs)

text_splitter= RecursiveCharacterTextSplitter(chunck_size=1000,chunk_overlap=20)
documents=text_splitter.split_documents(docs)
print(documents)


db=FAISS.from_documents(documents[:30],OllamaEmbeddings())



