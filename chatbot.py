import os
import time
import json
import numpy as np
from datetime import datetime

# Import data preprocessing modules
from modules.data_preprocessing.text_preprocessing import TextPreprocessor
from modules.data_preprocessing.audio_preprocessing import AudioPreprocessor
from modules.data_preprocessing.image_preprocessing import ImagePreprocessor

# Import emotion recognition modules
from modules.emotion_recognition.text_emotion_recognition import TextEmotionRecognizer
from modules.emotion_recognition.voice_emotion_recognition import VoiceEmotionRecognizer
from modules.emotion_recognition.facial_expression_recognition import FacialEmotionRecognizer

# Import fusion module
from modules.fusion.multimodal_fusion import MultimodalFusion

# Import needs assessment modules
from modules.needs_assessment.intent_recognition import IntentRecognizer
from modules.needs_assessment.severity_assessment import SeverityAssessor

# Import dialogue management module
from modules.dialogue_management.response_generation import ResponseGenerator

class EmotionalChatbot:
    """
    Main class for the multimodal emotionally intelligent chatbot.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the EmotionalChatbot with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize data preprocessing modules
        self.text_preprocessor = TextPreprocessor()
        self.audio_preprocessor = AudioPreprocessor(
            google_credentials_path=self.config.get('google_credentials_path')
        )
        self.image_preprocessor = ImagePreprocessor(
            google_credentials_path=self.config.get('google_credentials_path')
        )
        
        # Initialize emotion recognition modules
        self.text_emotion_recognizer = TextEmotionRecognizer(
            model_path=self.config.get('text_emotion_model_path')
        )
        self.voice_emotion_recognizer = VoiceEmotionRecognizer(
            model_path=self.config.get('voice_emotion_model_path')
        )
        self.facial_emotion_recognizer = FacialEmotionRecognizer(
            model_path=self.config.get('facial_emotion_model_path')
        )
        
        # Initialize fusion module
        self.emotion_fusion = MultimodalFusion(
            model_path=self.config.get('fusion_model_path')
        )
        
        # Initialize needs assessment modules
        self.intent_recognizer = IntentRecognizer(
            model_path=self.config.get('intent_model_path'),
            config_path=self.config.get('rasa_config_path')
        )
        self.severity_assessor = SeverityAssessor()
        
        # Initialize dialogue management module
        self.response_generator = ResponseGenerator(
            response_repository_path=self.config.get('response_repository_path')
        )
        
        # Initialize conversation context
        self.conversation_context = {
            'user_id': None,
            'session_id': self.generate_session_id(),
            'conversation_history': [],
            'detected_emotions': [],
            'detected_intents': [],
            'severity_history': [],
            'user_preferences': {},
            'start_time': datetime.now(),
            'last_activity': datetime.now()
        }
        
        print("Emotional Chatbot initialized successfully.")
    
    def load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'google_credentials_path': None,
            'text_emotion_model_path': None,
            'voice_emotion_model_path': None,
            'facial_emotion_model_path': None,
            'fusion_model_path': None,
            'intent_model_path': None,
            'rasa_config_path': None,
            'response_repository_path': None,
            'log_directory': 'logs',
            'data_directory': 'data',
            'use_multimodal': True,
            'use_text_emotion': True,
            'use_voice_emotion': True,
            'use_facial_emotion': True,
            'log_conversations': True,
            'crisis_resources': {
                'suicide_prevention_lifeline': '1-800-273-8255',
                'crisis_text_line': 'Text HOME to 741741',
                'emergency_services': '911'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update default config with loaded values
                default_config.update(loaded_config)
                print(f"Configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
        else:
            print("Using default configuration.")
        
        return default_config
    
    def generate_session_id(self):
        """
        Generate a unique session ID.
        
        Returns:
            str: Session ID
        """
        return f"session_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    def process_text_input(self, text):
        """
        Process text input to extract emotions and intent.
        
        Args:
            text (str): User's text input
            
        Returns:
            tuple: (preprocessed_text, text_emotions, intent_result)
        """
        # Preprocess text
        preprocessed_text = self.text_preprocessor.preprocess_for_emotion(text)
        
        # Recognize emotions from text
        text_emotions = None
        if self.config.get('use_text_emotion', True):
            text_emotions = self.text_emotion_recognizer.predict_emotion(preprocessed_text)
        
        # Recognize intent
        intent_result = self.intent_recognizer.predict_intent(text)
        
        return preprocessed_text, text_emotions, intent_result
    
    def process_voice_input(self, audio_file_path=None, audio_data=None):
        """
        Process voice input to extract emotions and convert to text.
        
        Args:
            audio_file_path (str): Path to audio file
            audio_data (numpy.ndarray): Audio data
            
        Returns:
            tuple: (transcribed_text, voice_emotions)
        """
        if not self.config.get('use_voice_emotion', True):
            return None, None
        
        # Load audio if file path is provided
        if audio_file_path and not audio_data:
            audio_data, sample_rate = self.audio_preprocessor.load_audio(audio_file_path)
        
        if audio_data is None:
            return None, None
        
        # Extract audio features
        features = self.audio_preprocessor.extract_features(audio_data)
        
        # Recognize emotions from voice
        voice_emotions = None
        if features and 'mfcc' in features:
            voice_emotions = self.voice_emotion_recognizer.predict_emotion(features['mfcc'])
        
        # Convert audio to text
        transcribed_text = self.audio_preprocessor.audio_to_text(
            audio_file_path=audio_file_path,
            audio_data=audio_data
        )
        
        return transcribed_text, voice_emotions
    
    def process_facial_input(self, image_path=None, image=None):
        """
        Process facial input to extract emotions.
        
        Args:
            image_path (str): Path to image file
            image (numpy.ndarray): Image data
            
        Returns:
            dict: Facial emotions
        """
        if not self.config.get('use_facial_emotion', True):
            return None
        
        # Load image if path is provided
        if image_path and not image:
            image = self.image_preprocessor.load_image(image_path)
        
        if image is None:
            return None
        
        # Detect faces
        faces = self.image_preprocessor.detect_faces(image)
        
        if not faces:
            return None
        
        # Process the first detected face
        face_region = faces[0]
        face_img = self.image_preprocessor.extract_face(image, face_region)
        
        if face_img is None:
            return None
        
        # Preprocess face
        processed_face = self.image_preprocessor.preprocess_face(face_img)
        
        # Recognize emotions from face
        facial_emotions = None
        if processed_face is not None:
            facial_emotions = self.facial_emotion_recognizer.predict_emotion(processed_face)
        
        return facial_emotions
    
    def fuse_emotions(self, text_emotions, voice_emotions, facial_emotions):
        """
        Fuse emotions from multiple modalities.
        
        Args:
            text_emotions (dict): Emotions detected from text
            voice_emotions (dict): Emotions detected from voice
            facial_emotions (dict): Emotions detected from facial expressions
            
        Returns:
            dict: Fused emotions
        """
        if not self.config.get('use_multimodal', True):
            # If multimodal fusion is disabled, prioritize text emotions
            return text_emotions or voice_emotions or facial_emotions
        
        # Prepare emotions dictionary for fusion
        emotions_dict = {}
        if text_emotions:
            emotions_dict['text'] = text_emotions
        if voice_emotions:
            emotions_dict['voice'] = voice_emotions
        if facial_emotions:
            emotions_dict['face'] = facial_emotions
        
        # If no emotions detected, return None
        if not emotions_dict:
            return None
        
        # If only one modality is available, return it directly
        if len(emotions_dict) == 1:
            return list(emotions_dict.values())[0]
        
        # Fuse emotions
        fused_emotions = self.emotion_fusion.fuse_emotions(emotions_dict)
        
        # If fusion fails, use rule-based fusion as fallback
        if fused_emotions is None:
            fused_emotions = self.emotion_fusion.rule_based_fusion(emotions_dict)
        
        return fused_emotions
    
    def assess_user_needs(self, text, emotions, intent_result):
        """
        Assess the user's needs based on text, emotions, and intent.
        
        Args:
            text (str): User's text input
            emotions (dict): Detected emotions
            intent_result (dict): Intent recognition result
            
        Returns:
            dict: Severity assessment
        """
        # Get dominant emotion
        dominant_emotion = None
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Assess severity
        severity_assessment = self.severity_assessor.assess_severity(
            text=text,
            emotions=emotions,
            intent_result=intent_result
        )
        
        return severity_assessment
    
    def generate_response(self, intent_result, emotions, severity_assessment):
        """
        Generate an appropriate response based on intent, emotions, and severity.
        
        Args:
            intent_result (dict): Intent recognition result
            emotions (dict): Detected emotions
            severity_assessment (dict): Severity assessment
            
        Returns:
            str: Generated response
        """
        # Extract intent name
        intent_name = intent_result['intent']['name']
        
        # Extract dominant emotion
        dominant_emotion = None
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Extract severity level
        severity_level = severity_assessment.get('severity_level', 'low')
        
        # Generate response
        response = self.response_generator.generate_response(
            intent=intent_name,
            emotion=dominant_emotion,
            severity=severity_level
        )
        
        return response
    
    def update_conversation_context(self, user_input, text_emotions, voice_emotions, facial_emotions, 
                                   fused_emotions, intent_result, severity_assessment, response):
        """
        Update the conversation context with new information.
        
        Args:
            user_input (str): User's input
            text_emotions (dict): Emotions detected from text
            voice_emotions (dict): Emotions detected from voice
            facial_emotions (dict): Emotions detected from facial expressions
            fused_emotions (dict): Fused emotions
            intent_result (dict): Intent recognition result
            severity_assessment (dict): Severity assessment
            response (str): Generated response
        """
        # Update last activity time
        self.conversation_context['last_activity'] = datetime.now()
        
        # Add to conversation history
        self.conversation_context['conversation_history'].append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to detected emotions
        if fused_emotions:
            self.conversation_context['detected_emotions'].append({
                'fused': fused_emotions,
                'text': text_emotions,
                'voice': voice_emotions,
                'facial': facial_emotions,
                'timestamp': datetime.now().isoformat()
            })
        
        # Add to detected intents
        if intent_result:
            self.conversation_context['detected_intents'].append({
                'intent': intent_result['intent']['name'],
                'confidence': intent_result['intent']['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Add to severity history
        if severity_assessment:
            self.conversation_context['severity_history'].append({
                'level': severity_assessment['severity_level'],
                'score': severity_assessment['severity_score'],
                'requires_attention': severity_assessment['requires_immediate_attention'],
                'timestamp': datetime.now().isoformat()
            })
    
    def log_conversation(self):
        """
        Log the conversation to a file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.config.get('log_conversations', True):
            return False
        
        try:
            # Create log directory if it doesn't exist
            log_dir = self.config.get('log_directory', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # Create log file name
            session_id = self.conversation_context['session_id']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"conversation_{session_id}_{timestamp}.json")
            
            # Write conversation context to file
            with open(log_file, 'w') as f:
                json.dump(self.conversation_context, f, indent=2)
            
            print(f"Conversation logged to {log_file}")
            return True
        except Exception as e:
            print(f"Error logging conversation: {e}")
            return False
    
    def process_input(self, text=None, audio_file_path=None, audio_data=None, image_path=None, image=None):
        """
        Process user input from multiple modalities.
        
        Args:
            text (str): Text input
            audio_file_path (str): Path to audio file
            audio_data (numpy.ndarray): Audio data
            image_path (str): Path to image file
            image (numpy.ndarray): Image data
            
        Returns:
            str: Generated response
        """
        # Process text input
        preprocessed_text = None
        text_emotions = None
        intent_result = None
        
        if text:
            preprocessed_text, text_emotions, intent_result = self.process_text_input(text)
        
        # Process voice input
        transcribed_text = None
        voice_emotions = None
        
        if audio_file_path or audio_data is not None:
            transcribed_text, voice_emotions = self.process_voice_input(audio_file_path, audio_data)
            
            # If no text input provided, use transcribed text
            if not text and transcribed_text:
                text = transcribed_text
                preprocessed_text, text_emotions, intent_result = self.process_text_input(text)
        
        # Process facial input
        facial_emotions = None
        
        if image_path or image is not None:
            facial_emotions = self.process_facial_input(image_path, image)
        
        # If no text input or transcription, return error message
        if not text:
            return "I'm sorry, I couldn't understand your input. Please try again."
        
        # Fuse emotions
        fused_emotions = self.fuse_emotions(text_emotions, voice_emotions, facial_emotions)
        
        # Assess user needs
        severity_assessment = self.assess_user_needs(text, fused_emotions, intent_result)
        
        # Generate response
        response = self.generate_response(intent_result, fused_emotions, severity_assessment)
        
        # Update conversation context
        self.update_conversation_context(
            user_input=text,
            text_emotions=text_emotions,
            voice_emotions=voice_emotions,
            facial_emotions=facial_emotions,
            fused_emotions=fused_emotions,
            intent_result=intent_result,
            severity_assessment=severity_assessment,
            response=response
        )
        
        return response
    
    def get_conversation_summary(self):
        """
        Get a summary of the conversation.
        
        Returns:
            dict: Conversation summary
        """
        # Calculate conversation duration
        start_time = self.conversation_context['start_time']
        last_activity = self.conversation_context['last_activity']
        duration = (last_activity - start_time).total_seconds()
        
        # Count turns
        turns = len(self.conversation_context['conversation_history'])
        
        # Get most frequent intent
        intents = [item['intent'] for item in self.conversation_context['detected_intents']]
        most_frequent_intent = max(set(intents), key=intents.count) if intents else None
        
        # Get most frequent emotion
        emotions = []
        for item in self.conversation_context['detected_emotions']:
            if item['fused']:
                dominant_emotion = max(item['fused'].items(), key=lambda x: x[1])[0]
                emotions.append(dominant_emotion)
        most_frequent_emotion = max(set(emotions), key=emotions.count) if emotions else None
        
        # Get highest severity
        severity_levels = [item['level'] for item in self.conversation_context['severity_history']]
        highest_severity = max(severity_levels, key=lambda x: {
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }.get(x, 0)) if severity_levels else None
        
        return {
            'session_id': self.conversation_context['session_id'],
            'duration_seconds': duration,
            'turns': turns,
            'most_frequent_intent': most_frequent_intent,
            'most_frequent_emotion': most_frequent_emotion,
            'highest_severity': highest_severity,
            'start_time': start_time.isoformat(),
            'last_activity': last_activity.isoformat()
        }
    
    def reset_conversation(self):
        """
        Reset the conversation context.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Log the current conversation before resetting
            self.log_conversation()
            
            # Reset conversation context
            self.conversation_context = {
                'user_id': self.conversation_context.get('user_id'),
                'session_id': self.generate_session_id(),
                'conversation_history': [],
                'detected_emotions': [],
                'detected_intents': [],
                'severity_history': [],
                'user_preferences': self.conversation_context.get('user_preferences', {}),
                'start_time': datetime.now(),
                'last_activity': datetime.now()
            }
            
            return True
        except Exception as e:
            print(f"Error resetting conversation: {e}")
            return False