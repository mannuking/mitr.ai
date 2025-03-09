import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import os
from streamlit_lottie import st_lottie
import requests
import json

# Function to load Lottie animations from a URL with error handling
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"An error occurred while loading the Lottie file: {e}")
        return None

# Load Lottie animation
lottie_ai = load_lottiefile("assets/QA.json")  # Ensure the file exists

# Display Lottie Animation
if lottie_ai:
    st_lottie(lottie_ai, height=300, key="ai_animation")
else:
    st.warning("Lottie animation could not be loaded.")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("app/style.css")

# Initialize Gemini API
genai.configure(api_key=st.secrets["gemini"]["api_key"])

# Function to extract text from the PDF document
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text_data = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():  # Skip empty pages
                text_data.extend(text.split('\n'))
        return text_data
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return []

# Function to load data from PDF
@st.cache_data
def load_data(pdf_path):
    if os.path.exists('chunks.pkl'):
        with open('chunks.pkl', 'rb') as f:
            return pickle.load(f)
    chunks = extract_text_from_pdf(pdf_path)
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    return chunks

# Function to create FAISS index
@st.cache_resource
def create_index(chunks, _embedding_model):
    embeddings = _embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

# Initialize the embedding model (cached)
@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to select relevant context using TF-IDF and cosine similarity
def select_relevant_context(question, chunks, top_k=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    chunk_vectors = vectorizer.fit_transform(chunks)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    return " ".join(selected_chunks)

# Function to generate answers using Gemini API
def generate_answer_gemini(question, context, history):
    model = genai.GenerativeModel('gemini-pro')
    
    chat = model.start_chat(history=[])
    
    prompt = f"""You are a helpful assistant knowledgeable about the Residential Tenancy Act (RTA) of Ontario. 
    Provide accurate and detailed answers based on the context provided.
    
    Context from the Residential Tenancy Act:
    {context}
    
    Conversation history:
    {history}
    
    User: {question}
    Assistant:"""
    
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

# Load models and data
embedding_model = load_models()
pdf_path = 'RTA.pdf'  # Ensure RTA.pdf is in the same directory as app.py
chunks = load_data(pdf_path)
index = create_index(chunks, embedding_model)

# Streamlit UI
st.title("Residential Tenancy Act Chatbot")
st.write("Ask questions about the Residential Tenancy Act and get instant answers.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Your question"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            # Select relevant context
            context = select_relevant_context(prompt, chunks)
            
            # Compile history (last 5 messages)
            history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-5:]])
            
            # Generate answer using Gemini
            answer = generate_answer_gemini(prompt, context, history)
            st.markdown(answer)
    
    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Add a button to clear the conversation
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Hide the Streamlit branding
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
