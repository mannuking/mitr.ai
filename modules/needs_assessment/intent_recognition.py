import os
import json
import numpy as np
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Trainer

class IntentRecognizer:
    """
    A class for recognizing user intents using Rasa NLU.
    """
    
    def __init__(self, model_path=None, config_path=None):
        """
        Initialize the IntentRecognizer with a pre-trained Rasa NLU model.
        
        Args:
            model_path (str): Path to a pre-trained Rasa NLU model directory
            config_path (str): Path to Rasa NLU configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.interpreter = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            try:
                self.interpreter = Interpreter.load(model_path)
                print(f"Loaded Rasa NLU model from {model_path}")
            except Exception as e:
                print(f"Error loading Rasa NLU model: {e}")
        else:
            print("No Rasa NLU model loaded. Use train_model() to train a new model.")
    
    def predict_intent(self, text):
        """
        Predict intent from text using the Rasa NLU model.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing intent prediction and confidence
        """
        if self.interpreter is None:
            print("No Rasa NLU model loaded. Cannot predict intent.")
            return self.fallback_intent_recognition(text)
        
        try:
            # Parse text with Rasa NLU
            result = self.interpreter.parse(text)
            
            # Extract intent information
            intent = {
                'name': result['intent']['name'],
                'confidence': result['intent']['confidence']
            }
            
            # Extract entities if available
            entities = result.get('entities', [])
            
            return {
                'intent': intent,
                'entities': entities
            }
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return self.fallback_intent_recognition(text)
    
    def fallback_intent_recognition(self, text):
        """
        Fallback intent recognition using keyword matching.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing intent prediction and confidence
        """
        text = text.lower()
        
        # Define intent keywords
        intent_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'goodbye': ['bye', 'goodbye', 'see you', 'talk to you later', 'farewell'],
            'thanks': ['thank', 'thanks', 'appreciate', 'grateful'],
            'help': ['help', 'assist', 'support', 'guidance', 'advice'],
            'affirm': ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'correct'],
            'deny': ['no', 'nope', 'not', 'don\'t', 'never'],
            'mood_great': ['good', 'great', 'wonderful', 'fantastic', 'amazing', 'happy', 'joy', 'excited'],
            'mood_unhappy': ['bad', 'sad', 'unhappy', 'depressed', 'terrible', 'awful', 'miserable', 'lonely'],
            'anxiety': ['anxious', 'nervous', 'worry', 'stress', 'panic', 'fear', 'tension'],
            'depression': ['depressed', 'hopeless', 'worthless', 'empty', 'numb', 'tired', 'exhausted'],
            'suicidal': ['suicide', 'kill myself', 'end my life', 'don\'t want to live', 'better off dead'],
            'resources': ['resource', 'information', 'contact', 'hotline', 'therapist', 'doctor'],
            'coping': ['cope', 'manage', 'deal', 'handle', 'strategy', 'technique', 'exercise', 'meditation'],
            'chitchat': ['weather', 'sports', 'news', 'movie', 'music', 'hobby', 'food']
        }
        
        # Calculate scores for each intent
        scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[intent] = score
        
        # Determine the most likely intent
        if scores:
            max_score = max(scores.values())
            max_intents = [intent for intent, score in scores.items() if score == max_score]
            
            # If multiple intents have the same score, prioritize certain intents
            priority_order = ['suicidal', 'depression', 'anxiety', 'mood_unhappy', 'help', 'resources', 'coping']
            for priority_intent in priority_order:
                if priority_intent in max_intents:
                    return {
                        'intent': {
                            'name': priority_intent,
                            'confidence': 0.6  # Moderate confidence for keyword-based matching
                        },
                        'entities': []
                    }
            
            # If no priority intent, just take the first one
            return {
                'intent': {
                    'name': max_intents[0],
                    'confidence': 0.6
                },
                'entities': []
            }
        
        # Default to chitchat if no intent is detected
        return {
            'intent': {
                'name': 'chitchat',
                'confidence': 0.3  # Low confidence for default intent
            },
            'entities': []
        }
    
    def train_model(self, training_data_path, output_path=None):
        """
        Train a new Rasa NLU model.
        
        Args:
            training_data_path (str): Path to training data file or directory
            output_path (str): Path to save the trained model
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if not self.config_path:
            print("No configuration file specified. Cannot train model.")
            return False
        
        try:
            # Load training data
            training_data = load_data(training_data_path)
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Create trainer
            trainer = Trainer(RasaNLUModelConfig(config))
            
            # Train model
            interpreter = trainer.train(training_data)
            
            # Save model
            if output_path:
                model_directory = trainer.persist(output_path)
                self.model_path = model_directory
                self.interpreter = interpreter
                print(f"Model trained and saved to {model_directory}")
            else:
                print("Model trained but not saved (no output path provided)")
            
            return True
        except Exception as e:
            print(f"Error training Rasa NLU model: {e}")
            return False
    
    def assess_severity(self, text, intent_result=None):
        """
        Assess the severity of the user's mental health needs based on intent and text.
        
        Args:
            text (str): Input text
            intent_result (dict): Intent prediction result (optional)
            
        Returns:
            dict: Dictionary containing severity assessment
        """
        if intent_result is None:
            intent_result = self.predict_intent(text)
        
        intent_name = intent_result['intent']['name']
        text = text.lower()
        
        # Define severity levels
        severity_levels = {
            'low': ['chitchat', 'greeting', 'goodbye', 'thanks', 'affirm', 'deny'],
            'medium': ['mood_unhappy', 'help', 'coping'],
            'high': ['anxiety', 'depression'],
            'critical': ['suicidal']
        }
        
        # Determine base severity from intent
        severity = 'low'
        for level, intents in severity_levels.items():
            if intent_name in intents:
                severity = level
                break
        
        # Check for critical keywords that might override the intent-based severity
        critical_keywords = [
            'suicide', 'kill myself', 'end my life', 'don\'t want to live', 'better off dead',
            'no reason to live', 'want to die', 'plan to kill', 'hurt myself'
        ]
        
        high_keywords = [
            'hopeless', 'worthless', 'can\'t go on', 'giving up', 'no future',
            'severe depression', 'severe anxiety', 'panic attack', 'trauma',
            'self-harm', 'cutting', 'hurting myself'
        ]
        
        medium_keywords = [
            'sad', 'anxious', 'stressed', 'worried', 'upset',
            'lonely', 'isolated', 'struggling', 'difficulty coping'
        ]
        
        # Check for critical keywords
        if any(keyword in text for keyword in critical_keywords):
            severity = 'critical'
        elif severity != 'critical' and any(keyword in text for keyword in high_keywords):
            severity = 'high'
        elif severity not in ['critical', 'high'] and any(keyword in text for keyword in medium_keywords):
            severity = 'medium'
        
        # Map severity to numeric scale
        severity_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        
        return {
            'severity_level': severity,
            'severity_score': severity_map[severity],
            'requires_immediate_attention': severity == 'critical',
            'intent': intent_name
        }