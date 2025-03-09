import os
import sys
import time
import json
import numpy as np
import streamlit as st
import sounddevice as sd
import cv2
from PIL import Image
import soundfile as sf
import threading
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Set page configuration
st.set_page_config(
    page_title="Emotional Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to initialize the chatbot
from chatbot import EmotionalChatbot

@st.cache_resource
def initialize_chatbot():
    return EmotionalChatbot(config_path=None)  # Changed class name here

# Audio recording functions
def record_audio(duration=5, sample_rate=22050):
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.write("Recording finished!")
    return recording, sample_rate

def save_audio(recording, sample_rate, file_path):
    sf.write(file_path, recording, sample_rate)
    return file_path

# Webcam functions
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Could not capture image")
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# Initialize the chatbot
chatbot = initialize_chatbot()

# Set up the sidebar
st.sidebar.title("Emotional Chatbot")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712010.png", width=100)
st.sidebar.markdown("---")

# Modal selection checkboxes
st.sidebar.subheader("Input Modalities")
use_text = st.sidebar.checkbox("Text Input", value=True)
use_voice = st.sidebar.checkbox("Voice Input", value=False)
use_face = st.sidebar.checkbox("Facial Expression", value=False)

# Audio settings
if use_voice:
    st.sidebar.subheader("Audio Settings")
    audio_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

# Debug mode
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Display conversation status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Conversation Status")

# Ensure session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'emotions' not in st.session_state:
    st.session_state.emotions = {'text': None, 'voice': None, 'face': None, 'fused': None}

if 'severity' not in st.session_state:
    st.session_state.severity = None

if 'conversation_active' not in st.session_state:
    st.session_state.conversation_active = False

# Conversation status indicators
conversation_status = st.sidebar.empty()
if st.session_state.conversation_active:
    conversation_status.success("Conversation Active")
else:
    conversation_status.info("No Active Conversation")

# Emotion display in sidebar
st.sidebar.subheader("Detected Emotions")
emotion_display = st.sidebar.empty()

# Severity display in sidebar
st.sidebar.subheader("Severity Assessment")
severity_display = st.sidebar.empty()

# Function to update emotion display
def update_emotion_display():
    if st.session_state.emotions['fused']:
        # Create a bar chart of emotions
        emotions = st.session_state.emotions['fused']
        fig, ax = plt.subplots(figsize=(5, 3))
        emotions_sorted = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
        ax.bar(emotions_sorted.keys(), emotions_sorted.values())
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        emotion_display.pyplot(fig)
    else:
        emotion_display.info("No emotions detected yet")

# Function to update severity display
def update_severity_display():
    if st.session_state.severity:
        severity_level = st.session_state.severity.get('severity_level', 'unknown')
        
        if severity_level == 'critical':
            severity_display.error(f"Severity: {severity_level.upper()}")
        elif severity_level == 'high':
            severity_display.warning(f"Severity: {severity_level.capitalize()}")
        elif severity_level == 'medium':
            severity_display.info(f"Severity: {severity_level.capitalize()}")
        else:
            severity_display.success(f"Severity: {severity_level.capitalize()}")
    else:
        severity_display.info("No severity assessment yet")

# Main app area
st.title("Multimodal Emotionally Intelligent Chatbot")
st.markdown("This chatbot uses text, voice, and facial expressions to provide personalized mental health support.")

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

with input_container:
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if use_text:
            user_input = st.text_input("Type your message:", key="text_input")
        else:
            user_input = None
            st.info("Text input disabled. Enable it in the sidebar.")
    
    with col2:
        if use_voice:
            if st.button("Record Voice"):
                with st.spinner("Recording..."):
                    # Create audio directory if it doesn't exist
                    os.makedirs('app/audio', exist_ok=True)
                    
                    # Record and save audio
                    recording, sample_rate = record_audio(duration=audio_duration)
                    audio_path = save_audio(recording, sample_rate, 'app/audio/input.wav')
                    
                    # Process with chatbot in debug mode
                    if debug_mode:
                        st.write(f"Audio saved to {audio_path}")
                        st.audio(audio_path)
        else:
            st.info("Voice input disabled")
    
    with col3:
        if use_face:
            if st.button("Capture Face"):
                with st.spinner("Capturing..."):
                    # Create image directory if it doesn't exist
                    os.makedirs('app/images', exist_ok=True)
                    
                    # Capture and save image
                    image = capture_image()
                    if image is not None:
                        # Save the image
                        image_pil = Image.fromarray(image)
                        image_path = 'app/images/face.jpg'
                        image_pil.save(image_path)
                        
                        # Display in debug mode
                        if debug_mode:
                            st.write(f"Image saved to {image_path}")
                            st.image(image, caption="Captured Image", width=300)
        else:
            st.info("Facial input disabled")

# Process user input when submitted
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Set conversation as active
    st.session_state.conversation_active = True
    
    # Process the input with the chatbot
    audio_path = 'app/audio/input.wav' if use_voice and os.path.exists('app/audio/input.wav') else None
    image_path = 'app/images/face.jpg' if use_face and os.path.exists('app/images/face.jpg') else None
    
    # Debug information
    if debug_mode:
        st.write(f"Processing - Text: {bool(user_input)}, Audio: {bool(audio_path)}, Image: {bool(image_path)}")
    
    # Process the input
    with st.spinner("Thinking..."):
        # Use our chatbot to process the input
        response = chatbot.process_input(
            text=user_input,
            audio_file_path=audio_path,
            image_path=image_path
        )
        
        # Get emotions from chatbot's conversation context
        if chatbot.conversation_context['detected_emotions']:
            latest_emotions = chatbot.conversation_context['detected_emotions'][-1]
            st.session_state.emotions = {
                'text': latest_emotions.get('text'),
                'voice': latest_emotions.get('voice'),
                'facial': latest_emotions.get('fused')
            }
        
        # Get severity from conversation context
        if chatbot.conversation_context['severity_history']:
            latest_severity = chatbot.conversation_context['severity_history'][-1]
            st.session_state.severity = {
                'severity_level': latest_severity.get('level'),
                'severity_score': latest_severity.get('score'),
                'requires_immediate_attention': latest_severity.get('requires_attention')
            }
    
    # Add bot response to chat
    st.session_state.messages.append({"role": "bot", "content": response})
    
    # Clear the text input
    st.experimental_rerun()

# Update displays
update_emotion_display()
update_severity_display()

# Debug information
if debug_mode:
    st.markdown("---")
    st.subheader("Debug Information")
    
    # Emotion details
    if st.session_state.emotions['fused']:
        st.write("Detected Emotions:")
        st.json(st.session_state.emotions)
    
    # Severity details
    if st.session_state.severity:
        st.write("Severity Assessment:")
        st.json(st.session_state.severity)
    
    # Conversation summary
    if st.session_state.conversation_active:
        st.write("Conversation Summary:")
        summary = chatbot.get_conversation_summary()
        st.json(summary)

# Functionality to reset the conversation
st.sidebar.markdown("---")
if st.sidebar.button("Reset Conversation"):
    # Reset the chatbot
    chatbot.reset_conversation()
    
    # Reset the session state
    st.session_state.messages = []
    st.session_state.emotions = {'text': None, 'voice': None, 'face': None, 'fused': None}
    st.session_state.severity = None
    st.session_state.conversation_active = False
    
    # Rerun to update the UI
    st.experimental_rerun()
