import streamlit as st
import sqlite3
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

apikey = os.getenv('api_key')  # Retrieve the API key from environment variables


# Load pre-trained sentence embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index('doj_index.faiss')

# Connect to SQLite database
conn = sqlite3.connect('doj_data.db')
c = conn.cursor()

# Retrieve chunks from the database
c.execute('SELECT chunk FROM pdf_chunks')
chunked_text = [row[0] for row in c.fetchall()]

# Tokenize chunks for BM25
tokenized_chunks = [chunk.split() for chunk in chunked_text]
bm25 = BM25Okapi(tokenized_chunks)

def retrieve_semantic_chunks(query, top_n=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_n)
    return [chunked_text[i] for i in indices[0]]

def retrieve_chunks(query, top_n=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [chunked_text[i] for i in top_n_indices]

def generate_response(query):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_semantic_chunks(query)
    relevant_document = ' '.join(relevant_chunks)

    # Generate prompt
    prompt = f"""
    Remember you are a robot chatbot, do not add any extra information than what is given in the document.
    This is the recommended activity: {relevant_document}
    The user input is: {query}
    Users will give you mcq questions and you have to give the answer from the options.
    """
    key = apikey
   
    client = Groq(api_key=key)
    completion = client.chat.completions.create(
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=0.65,
        stream=True,
        stop=None,
    )

    full_response = []
    for chunk in completion:
        if chunk.choices[0].delta.content:
            full_response.append(chunk.choices[0].delta.content)

    return ''.join(full_response)

# Streamlit UI
st.title("?")

query = st.text_input("Ask:")
if query:
    response = generate_response(query)
    st.write("Response:", response)
