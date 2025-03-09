import os
# Set tokenizers parallelism and CUDA settings at the very start
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import numpy as np
import streamlit as st
import sounddevice as sd
import cv2
from PIL import Image
import soundfile as sf
import matplotlib.pyplot as plt
import asyncio
import google.generativeai as genai
from therapist_chatbot import display_therapist_chatbot

# Fix for event loop
def init_asyncio_patch():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

init_asyncio_patch()

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure Gemini API Key
genai.configure(api_key=st.secrets["gemini"]["api_key"])

# Function to initialize the chatbot
@st.cache_resource
def initialize_gemini_model():
    return genai.GenerativeModel('gemini-2.0-flash-lite')

# Function to generate answers using Gemini API
def generate_answer(text=None, audio_file_path=None, image_path=None):
    # Get inputs based on enabled modalities
    inputs = []
    
    if text and 'use_text' in st.session_state and st.session_state.use_text:
        inputs.append(f"Text Input: {text}")
    
    if audio_file_path and 'use_voice' in st.session_state and st.session_state.use_voice:
        # In a real implementation, you would transcribe audio here
        inputs.append(f"Audio Input: Audio was provided.")
    
    if image_path and 'use_face' in st.session_state and st.session_state.use_face:
        # In a real implementation, you would analyze the image here
        inputs.append("Image Input: Image was provided.")
    
    if not inputs:
        return "No input provided."
    
    prompt = "\n".join(inputs)
    
    prompt = f"""You are a helpful emotional support chatbot. Respond to the user's input,
    considering all provided modalities (text, audio, image).
    
    User Input:
    {prompt}
    
    Assistant:"""
    
    try:
        # Use the chat from session state
        response = st.session_state.chat.send_message(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

# Set page configuration
st.set_page_config(
    page_title="AI Therapist Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Audio recording functions
def record_audio(duration=5, sample_rate=22050):
    try:
        st.write("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        st.write("Recording finished!")
        return recording, sample_rate
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None, None

def save_audio(recording, sample_rate, file_path):
    try:
        if recording is not None:
            sf.write(file_path, recording, sample_rate)
            return file_path
        return None
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return None

# Webcam functions
def capture_image():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Could not capture image")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    except Exception as e:
        st.error(f"Error capturing image: {e}")
        return None

# Initialize the model
model = initialize_gemini_model()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'conversation_active' not in st.session_state:
    st.session_state.conversation_active = False
    
if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    
if 'use_text' not in st.session_state:
    st.session_state.use_text = True
    
if 'use_voice' not in st.session_state:
    st.session_state.use_voice = False
    
if 'use_face' not in st.session_state:
    st.session_state.use_face = False
    
if 'audio_duration' not in st.session_state:
    st.session_state.audio_duration = 5
    
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
    
if 'chatbot_mode' not in st.session_state:
    st.session_state.chatbot_mode = "general"  # Default to general chatbot

# Set up the sidebar
st.sidebar.title("AI Therapist Chatbot")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712010.png", width=100)
st.sidebar.markdown("---")

# Chatbot mode selection
st.sidebar.subheader("Chatbot Mode")
chatbot_mode = st.sidebar.radio(
    "Select Chatbot Mode",
    ["General Assistant", "Therapy Assistant"],
    index=0 if st.session_state.chatbot_mode == "general" else 1
)
st.session_state.chatbot_mode = "general" if chatbot_mode == "General Assistant" else "therapy"

# Modal selection checkboxes (only show for general mode)
if st.session_state.chatbot_mode == "general":
    st.sidebar.subheader("Input Modalities")
    st.session_state.use_text = st.sidebar.checkbox("Text Input", value=st.session_state.use_text)
    st.session_state.use_voice = st.sidebar.checkbox("Voice Input", value=st.session_state.use_voice)
    st.session_state.use_face = st.sidebar.checkbox("Facial Expression", value=st.session_state.use_face)

    # Audio settings
    if st.session_state.use_voice:
        st.sidebar.subheader("Audio Settings")
        st.session_state.audio_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, st.session_state.audio_duration)

    # Debug mode
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)

    # Display conversation status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Conversation Status")
    conversation_status = st.sidebar.empty()
    if st.session_state.conversation_active:
        conversation_status.success("Conversation Active")
    else:
        conversation_status.info("No Active Conversation")

# Main app area
if st.session_state.chatbot_mode == "general":
    st.title("AI Therapist Chatbot")
    st.markdown("This chatbot provides support and information based on a vast knowledge base.")

    # Create chat interface
    st.subheader("Chat")
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Bot:** {message['content']}")

    # Input area
    st.markdown("---")
    input_container = st.container()

    # Create a form for the input to prevent automatic resubmission
    with st.form(key="input_form"):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if st.session_state.use_text:
                user_input = st.text_input("Type your message:", key="text_input")
            else:
                user_input = None
                st.info("Text input disabled. Enable it in the sidebar.")
        
        # Submit button for the form
        submit_button = st.form_submit_button("Send")

    # Voice and face capture buttons outside the form
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state.use_voice:
            if st.button("Record Voice"):
                with st.spinner("Recording..."):
                    os.makedirs('app/audio', exist_ok=True)
                    recording, sample_rate = record_audio(duration=st.session_state.audio_duration)
                    audio_path = save_audio(recording, sample_rate, 'app/audio/input.wav')
                    if st.session_state.debug_mode and audio_path:
                        st.write(f"Audio saved to {audio_path}")
                        st.audio(audio_path)
                        
                    # Process voice input immediately
                    if audio_path:
                        with st.spinner("Processing voice input..."):
                            response = generate_answer(audio_file_path=audio_path)
                            # Add a placeholder message for the voice input
                            st.session_state.messages.append({"role": "user", "content": "[Voice Input]"})
                            st.session_state.messages.append({"role": "bot", "content": response})
                            st.session_state.conversation_active = True
                            st.rerun()

    with col2:
        if st.session_state.use_face:
            if st.button("Capture Face"):
                with st.spinner("Capturing..."):
                    os.makedirs('app/images', exist_ok=True)
                    image = capture_image()
                    if image is not None:
                        image_pil = Image.fromarray(image)
                        image_path = 'app/images/face.jpg'
                        image_pil.save(image_path)
                        if st.session_state.debug_mode:
                            st.write(f"Image saved to {image_path}")
                            st.image(image, caption="Captured Image", width=300)
                        
                        # Process facial input immediately
                        if image_path:
                            with st.spinner("Processing facial expression..."):
                                response = generate_answer(image_path=image_path)
                                # Add a placeholder message for the facial input
                                st.session_state.messages.append({"role": "user", "content": "[Facial Expression]"})
                                st.session_state.messages.append({"role": "bot", "content": response})
                                st.session_state.conversation_active = True
                                st.rerun()

    # Process text input when submitted
    if submit_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_active = True
        
        # Check for additional modalities
        audio_path = 'app/audio/input.wav' if st.session_state.use_voice and os.path.exists('app/audio/input.wav') else None
        image_path = 'app/images/face.jpg' if st.session_state.use_face and os.path.exists('app/images/face.jpg') else None
        
        if st.session_state.debug_mode:
            st.write(f"Processing - Text: {bool(user_input)}, Audio: {bool(audio_path)}, Image: {bool(image_path)}")
        
        # Process the input
        with st.spinner("Thinking..."):
            response = generate_answer(text=user_input, audio_file_path=audio_path, image_path=image_path)
            st.session_state.messages.append({"role": "bot", "content": response})
        
        # Rerun to update the UI
        st.rerun()

else:
    # Display the therapy chatbot
    display_therapist_chatbot()

# Reset conversation button
st.sidebar.markdown("---")
if st.sidebar.button("Reset Conversation"):
    # Reset the chat
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []
    st.session_state.conversation_active = False
    st.rerun()
