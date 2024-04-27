from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,Chroma 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
#Data Ingestion
loader=PyPDFLoader("transformer.pdf")
document=loader.load()
#print(document)

#splitting the text
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.split_documents(document)




#Embeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
   api_key="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", model_name="sentence-transformers/all-MiniLM-l6-v2")

#query_result = hf_embeddings.embed_query(chunks[0].page_content)
#print(query_result)



# vector database

vector_db=FAISS.from_documents(documents=chunks[:200],embedding=hf_embeddings)

#query with similarity search
query="describe the training regime for our models?"
relevant_doc=vector_db.similarity_search(query)
#print(relevant_doc[0].page_content)


retriever=vector_db.as_retriever(search_type="similarity",search_kwargs={"k":3})
print(retriever)

#huggingface-hub

from langchain_community.llms import HuggingFaceHub
hf=HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature":0.1,"max_length":500}
)




prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{question}

Helpful Answers:
 """



prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}

)


questions="what are the detail of Hardware and Schedule"

# Call the QA chain with our query.
result = retrievalQA.invoke({"query": questions})
print(result['result'])