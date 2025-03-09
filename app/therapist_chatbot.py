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
import json

# Define default therapy text
DEFAULT_THERAPY_TEXT = [
    "Therapy is a collaborative treatment based on the relationship between an individual and a therapist.",
    "Common types of therapy include cognitive behavioral therapy (CBT), psychodynamic therapy, and humanistic therapy.",
    "CBT focuses on identifying and changing negative thought patterns and behaviors.",
    "Mindfulness techniques can help reduce anxiety and stress by focusing on the present moment.",
    "The therapeutic alliance between client and therapist is crucial for successful treatment outcomes.",
    "Emotional regulation skills help people manage and respond to emotional experiences effectively.",
    "Mental health is as important as physical health and should be treated with equal care and attention.",
    "Therapy provides a safe, confidential space to explore feelings, behaviors, and experiences.",
    "Self-care practices are essential for maintaining good mental health and emotional well-being.",
    "Depression and anxiety are common mental health conditions that can be effectively treated with therapy."
]

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
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Use relative path since both files are in the same directory
local_css("style.css")

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

# Function to combine text from multiple PDFs
def load_multiple_pdfs(pdf_paths):
    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            chunks = extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {e}")
    return all_chunks

# Function to load data from PDF
@st.cache_data
def load_data(pdf_paths):
    try:
        if os.path.exists('chunks.pkl'):
            with open('chunks.pkl', 'rb') as f:
                chunks = pickle.load(f)
                if chunks:  # Verify we have valid data
                    return chunks
        
        all_chunks = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                st.warning(f"PDF file not found: {pdf_path}")
                continue
                
            chunks = extract_text_from_pdf(pdf_path)
            if chunks:
                all_chunks.extend(chunks)
        
        if not all_chunks:  # If no PDFs were successfully loaded
            st.warning("No PDF content loaded. Using default therapy information.")
            all_chunks = DEFAULT_THERAPY_TEXT
            
        with open('chunks.pkl', 'wb') as f:
            pickle.dump(all_chunks, f)
        
        return all_chunks
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return DEFAULT_THERAPY_TEXT

# Function to create FAISS index
@st.cache_resource
def create_index(chunks, _embedding_model):
    try:
        if not chunks:
            st.error("No text chunks available for indexing")
            return None
            
        embeddings = _embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        if embeddings.size == 0:
            st.error("No embeddings were generated")
            return None
            
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        return None

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
    model = genai.GenerativeModel('gemini-2.0-flash-lite')  # Updated model name
    
    chat = model.start_chat(history=[])
    
    prompt = f"""You are a helpful and empathetic therapy assistant with expertise in counseling and mental health. 
    Provide accurate, ethical, and supportive answers based on the context provided from therapy books.
    
    Context from therapy books:
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

def display_therapist_chatbot():
    try:
        # Load models and data
        embedding_model = load_models()
        pdf_paths = [
            "data/The Gift of Therapy_ An Open Letter to a New Generation of -- Yalom, Irvin -- 2013 -- Harper Perennial --.pdf",
            "data/Theory_and_Practice_of_Counseling_a(b-ok.org) (1).pdf"
        ]
        
        # Load chunks
        chunks = load_data(pdf_paths)
        if not chunks:
            st.error("Failed to load any text chunks")
            return
        
        # Create index
        index = create_index(chunks, embedding_model)
        if not index:
            st.error("Failed to create search index")
            return
        
        # Streamlit UI
        st.title("AI Therapy Assistant")
        st.write("Ask questions about therapy, counseling, and mental health to get instant answers.")

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

    except Exception as e:
        st.error(f"Application initialization error: {e}")
        return

# Only run the app directly if this file is run directly
if __name__ == "__main__":
    display_therapist_chatbot()
