import streamlit as st
import requests

def get_openai_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke")
    json={'input':{'topic':input_text}}
    return response.json()['output']['content']


st.title("Demo of API using langchain")
input_text=st.text_input("Write a essay on")

if input_text:
    st.write(get_openai_response(input_text))