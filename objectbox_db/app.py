import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings,OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()


os.environ['HUGGINGFACE_API_KEY']= os.getenv("HUGGINGFACE_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.title("ObjectBox VectorDB using the llama3")

llm=ChatGroq(api_key=groq_api_key,model="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)

def vector_embeddings():
    if "vector" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("GenAI_projects\objectbox_db\Vijayawada to Mumbai Tickets  1.pdf")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=768)

vector_embeddings()

input_prompt=st.text_input("enter your queries")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("ObjectBox Database is ready")


if input_prompt:
    docs_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vector.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,docs_chain)
    start=time.process_time()

    response=retrieval_chain.invoke({'input':input_prompt})

    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


