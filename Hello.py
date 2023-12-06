import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import math
import chromadb
from chromadb.utils import embedding_functions
import csv
import uuid
#openpyxl
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

data = pd.read_excel('Dr Mix Chatbot Q&A.xlsx', sheet_name='Sheet1')
data_x=data[['Question','Answer']].drop_duplicates().to_dict('records')
#print(data.head())

questions = []
questions.extend(data['Question'])
#print(len(questions))

# Remove 'nan' values
nan = "nan"
questions = [item for item in questions if not isinstance(item, float) or not math.isnan(item)]

# Print the cleaned list
#print(questions)

id_lst = [uuid.uuid4().hex for i in range(0,len(questions))]
#print(id_lst)


huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="sentence-transformers/multi-qa-distilbert-cos-v1"
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="dr_questions_data")
#print(collection)

collection.add(
    documents=questions,
    ids=id_lst
)

def chat_engine_response(search_query,vector_db,stored_data_dict):
    results = collection.query(query_texts=[user_prompt],n_results=1)
    question = results['documents'][0][0]
    for row in stored_data_dict:
        if row['Question'] == question:
            response = row['Answer']
            return response

st.title("Dr Mix Virtual Assitant")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Dr Mix, how can I help you"}]


# Display chat messages to screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# get user prompt
user_prompt = st.chat_input()

# check is user typed in something
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

# Get the last message, if not AI(assistant), generate LLM response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = chat_engine_response(user_prompt,collection,data_x)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
