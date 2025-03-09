import os
import json
import random
import numpy as np

class ResponseGenerator:
    """
    A class for generating appropriate responses based on user intent, emotion, and severity assessment.
    """
    
    def __init__(self, response_repository_path=None):
        """
        Initialize the ResponseGenerator with a response repository.
        
        Args:
            response_repository_path (str): Path to the response repository JSON file
        """
        self.response_repository = {}
        
        # Load response repository if path is provided
        if response_repository_path and os.path.exists(response_repository_path):
            try:
                with open(response_repository_path, 'r') as f:
                    self.response_repository = json.load(f)
                print(f"Loaded response repository from {response_repository_path}")
            except Exception as e:
                print(f"Error loading response repository: {e}")
                self.initialize_default_repository()
        else:
            print("No response repository loaded. Using default responses.")
            self.initialize_default_repository()
    
    def initialize_default_repository(self):
        """
        Initialize a default response repository.
        """
        self.response_repository = {
            # Greetings
            'greeting': {
                'default': [
                    "Hello! I'm here to support you. How are you feeling today?",
                    "Hi there! I'm your mental health support bot. How can I help you today?",
                    "Hello! I'm here to listen and help. How are you doing?"
                ],
                'joy': [
                    "Hello! You seem to be in good spirits today. That's wonderful! How can I help you?",
                    "Hi there! I'm glad to see you're doing well. What brings you here today?"
                ],
                'sadness': [
                    "Hello. I notice you might be feeling down. I'm here to listen and support you.",
                    "Hi there. I'm here for you during difficult times. Would you like to talk about what's bothering you?"
                ],
                'anger': [
                    "Hello. I can see you might be frustrated. It's okay to feel that way. Would you like to talk about it?",
                    "Hi there. I'm here to help with whatever is bothering you. Would you like to share what's on your mind?"
                ],
                'fear': [
                    "Hello. I'm here to support you through any worries or concerns you might have.",
                    "Hi there. I understand you might be feeling anxious. I'm here to help you through it."
                ]
            },
            
            # Goodbyes
            'goodbye': {
                'default': [
                    "Take care! Remember I'm here whenever you need to talk.",
                    "Goodbye for now. Remember to be kind to yourself.",
                    "Take care of yourself. I'll be here if you need support in the future."
                ]
            },
            
            # Thanks
            'thanks': {
                'default': [
                    "You're welcome! I'm glad I could help.",
                    "It's my pleasure to support you.",
                    "I'm here for you anytime you need to talk."
                ]
            },
            
            # Help
            'help': {
                'default': [
                    "I'm here to provide emotional support and resources. What specifically are you looking for help with?",
                    "I can help with emotional support, coping strategies, or connecting you with resources. What would be most helpful right now?"
                ],
                'high': [
                    "I understand you're going through a difficult time. Let's focus on what might help you right now. Would you like to talk about some coping strategies or resources?",
                    "I'm here to support you through this challenging time. Would it help to discuss some immediate coping techniques or resources for professional support?"
                ],
                'critical': [
                    "I'm concerned about what you're sharing. It's important to connect with a mental health professional right away. Would you like information on crisis resources?",
                    "What you're experiencing sounds serious. Let me help you find immediate support. There are professionals available 24/7 who can help you through this."
                ]
            },
            
            # Mood - Great
            'mood_great': {
                'default': [
                    "I'm glad to hear you're doing well! What's been going well for you?",
                    "That's wonderful to hear! What positive things have been happening in your life?"
                ]
            },
            
            # Mood - Unhappy
            'mood_unhappy': {
                'default': [
                    "I'm sorry to hear you're feeling down. Would you like to talk about what's bothering you?",
                    "It sounds like you're having a difficult time. I'm here to listen if you'd like to share more."
                ],
                'medium': [
                    "I understand you're feeling sad. That's a normal emotion, but it can be challenging. Would you like to explore some coping strategies?",
                    "I'm sorry you're feeling this way. Would it help to talk about what might be contributing to these feelings?"
                ],
                'high': [
                    "I'm really sorry you're feeling this way. These feelings can be overwhelming, but you don't have to face them alone. Would it help to discuss some resources or support options?",
                    "Thank you for sharing how you're feeling. It takes courage to acknowledge these emotions. Would you like to talk about some strategies that might help?"
                ]
            },
            
            # Anxiety
            'anxiety': {
                'default': [
                    "It sounds like you're feeling anxious. Would you like to try a simple breathing exercise to help calm your mind?",
                    "Anxiety can be really challenging. Would it help to talk about what's causing your anxiety?"
                ],
                'medium': [
                    "I understand anxiety can be difficult to manage. Would you like to try a grounding technique that might help in the moment?",
                    "Anxiety is a common experience, but that doesn't make it any easier. Would you like to explore some coping strategies?"
                ],
                'high': [
                    "I hear that you're experiencing significant anxiety. This can feel overwhelming, but there are techniques that might help. Would you like to try a guided breathing exercise?",
                    "When anxiety is intense, it can be helpful to focus on the present moment. Would you like to try a brief mindfulness exercise?"
                ]
            },
            
            # Depression
            'depression': {
                'default': [
                    "I'm sorry you're feeling this way. Depression can make everything feel more difficult. Would you like to talk about it?",
                    "Thank you for sharing how you're feeling. Would it help to discuss some strategies that might provide some relief?"
                ],
                'medium': [
                    "Depression can make it hard to see a way forward, but there are small steps that can help. Would you like to talk about some of these?",
                    "I understand these feelings can be overwhelming. Would it help to discuss some self-care strategies or resources?"
                ],
                'high': [
                    "I'm really sorry you're experiencing these difficult feelings. It's important to know that help is available. Would you like information about professional support options?",
                    "Thank you for trusting me with these feelings. Depression can be isolating, but you're not alone. Would it help to talk about connecting with professional support?"
                ],
                'critical': [
                    "I'm concerned about what you're sharing. It's really important to connect with a mental health professional. Would you like information on how to do that?",
                    "What you're experiencing sounds serious. Let me help you find support. There are professionals who specialize in helping people through these exact feelings."
                ]
            },
            
            # Suicidal
            'suicidal': {
                'default': [
                    "I'm very concerned about what you're sharing. Your life matters, and there are people who can help you through this difficult time. The National Suicide Prevention Lifeline is available 24/7 at 1-800-273-8255. Would you be willing to call them now?",
                    "Thank you for trusting me with these thoughts. It's really important that you talk to a crisis counselor who can provide immediate support. The Crisis Text Line is available 24/7 - just text HOME to 741741. Would you be willing to reach out to them?"
                ],
                'critical': [
                    "I'm very concerned about your safety right now. Please call the National Suicide Prevention Lifeline at 1-800-273-8255 immediately. They have trained counselors available 24/7 who can help you through this crisis.",
                    "What you're experiencing requires immediate attention from a crisis professional. Please text HOME to 741741 to reach the Crisis Text Line, or call 1-800-273-8255 for the National Suicide Prevention Lifeline. Would you like me to provide more crisis resources?"
                ]
            },
            
            # Resources
            'resources': {
                'default': [
                    "Here are some mental health resources that might be helpful:\n- National Alliance on Mental Illness (NAMI): nami.org\n- Mental Health America: mentalhealthamerica.net\n- Psychology Today Therapist Finder: psychologytoday.com/us/therapists\nIs there a specific type of resource you're looking for?",
                    "I'd be happy to share some resources. Here are a few options:\n- National Institute of Mental Health: nimh.nih.gov\n- Anxiety and Depression Association of America: adaa.org\n- MentalHealth.gov: mentalhealth.gov\nWould you like more specific resources?"
                ],
                'high': [
                    "Given what you've shared, here are some resources that might be particularly helpful:\n- National Suicide Prevention Lifeline: 1-800-273-8255\n- Crisis Text Line: Text HOME to 741741\n- SAMHSA Treatment Locator: findtreatment.samhsa.gov\nWould you like me to provide more information about any of these?",
                    "Based on our conversation, I think these resources might be helpful:\n- National Helpline: 1-800-662-HELP (4357)\n- Psychology Today Therapist Finder: psychologytoday.com/us/therapists\n- Open Path Collective (affordable therapy): openpathcollective.org\nIs there a specific type of support you're looking for?"
                ],
                'critical': [
                    "It's important that you connect with crisis support right away. Here are immediate resources:\n- National Suicide Prevention Lifeline: 1-800-273-8255\n- Crisis Text Line: Text HOME to 741741\n- Emergency Services: 911\nWould you be willing to reach out to one of these services now?",
                    "Based on what you've shared, I strongly recommend connecting with crisis support. Here are resources available 24/7:\n- National Suicide Prevention Lifeline: 1-800-273-8255\n- Crisis Text Line: Text HOME to 741741\n- Find your local crisis center: suicidepreventionlifeline.org/our-crisis-centers/\nCan I help you connect with any of these resources?"
                ]
            },
            
            # Coping
            'coping': {
                'default': [
                    "There are many coping strategies that can help with difficult emotions. Some options include deep breathing, mindfulness, physical activity, or creative expression. Would you like to explore any of these further?",
                    "Coping strategies can be very personal - what works for one person might not work for another. Some possibilities include journaling, meditation, talking with friends, or engaging in hobbies. Is there a particular area you'd like to focus on?"
                ],
                'anxiety': [
                    "For anxiety, some helpful coping strategies include deep breathing, progressive muscle relaxation, and grounding techniques. Would you like me to guide you through one of these?",
                    "When dealing with anxiety, it can help to focus on the present moment. The 5-4-3-2-1 technique involves naming 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. Would you like to try this or explore other anxiety management techniques?"
                ],
                'depression': [
                    "For depression, some helpful strategies include gentle physical activity, maintaining social connections, and establishing a routine. Small steps can make a difference. Would you like to discuss any of these approaches?",
                    "When dealing with depression, self-care becomes especially important. This might include ensuring you're getting adequate sleep, nutrition, and some form of physical movement. Would you like to talk about creating a self-care plan?"
                ]
            },
            
            # Chitchat
            'chitchat': {
                'default': [
                    "I'm happy to chat! What would you like to talk about?",
                    "I'm here for both serious conversations and casual chats. What's on your mind?"
                ]
            },
            
            # Affirm
            'affirm': {
                'default': [
                    "Great! Let's continue.",
                    "Excellent. Let's move forward."
                ]
            },
            
            # Deny
            'deny': {
                'default': [
                    "That's okay. What would you prefer to talk about?",
                    "No problem. What would be more helpful for you right now?"
                ]
            },
            
            # Fallback
            'fallback': {
                'default': [
                    "I'm not sure I understand. Could you please rephrase that?",
                    "I'm still learning and didn't quite catch that. Could you say it differently?",
                    "I'm sorry, I didn't understand. Could you explain in a different way?"
                ]
            }
        }
    
    def generate_response(self, intent, emotion=None, severity=None):
        """
        Generate an appropriate response based on intent, emotion, and severity.
        
        Args:
            intent (str): User intent
            emotion (str): Detected emotion (optional)
            severity (str): Severity level (optional)
            
        Returns:
            str: Generated response
        """
        # Get responses for the intent
        intent_responses = self.response_repository.get(intent)
        
        # If intent not found, use fallback
        if not intent_responses:
            intent_responses = self.response_repository.get('fallback')
        
        # If severity is critical, prioritize critical responses
        if severity == 'critical' and 'critical' in intent_responses:
            responses = intent_responses['critical']
        # If severity is high, prioritize high responses
        elif severity == 'high' and 'high' in intent_responses:
            responses = intent_responses['high']
        # If severity is medium, prioritize medium responses
        elif severity == 'medium' and 'medium' in intent_responses:
            responses = intent_responses['medium']
        # If emotion is provided and exists in the repository, use emotion-specific responses
        elif emotion and emotion in intent_responses:
            responses = intent_responses[emotion]
        # Otherwise, use default responses
        else:
            responses = intent_responses.get('default', ["I'm here to help. How can I support you today?"])
        
        # Select a random response from the appropriate category
        response = random.choice(responses)
        
        return response
    
    def generate_response_with_context(self, intent, emotion=None, severity=None, context=None):
        """
        Generate a response with additional context.
        
        Args:
            intent (str): User intent
            emotion (str): Detected emotion (optional)
            severity (str): Severity level (optional)
            context (dict): Additional context information (optional)
            
        Returns:
            str: Generated response
        """
        # Generate base response
        response = self.generate_response(intent, emotion, severity)
        
        # Add context-specific information if provided
        if context:
            # Add user name if available
            if 'user_name' in context:
                response = response.replace("Hello!", f"Hello {context['user_name']}!")
                response = response.replace("Hi there!", f"Hi {context['user_name']}!")
            
            # Add time-specific greeting if available
            if 'time_of_day' in context:
                time_of_day = context['time_of_day']
                if "Hello" in response:
                    response = response.replace("Hello", f"Good {time_of_day}")
                elif "Hi there" in response:
                    response = response.replace("Hi there", f"Good {time_of_day}")
            
            # Add previous topics if available
            if 'previous_topics' in context and intent == 'greeting' and random.random() < 0.5:
                topics = context['previous_topics']
                if topics:
                    last_topic = topics[-1]
                    response += f" Last time we talked about {last_topic}. Would you like to continue that conversation or discuss something new?"
        
        return response
    
    def add_response(self, intent, category, responses):
        """
        Add new responses to the repository.
        
        Args:
            intent (str): Intent category
            category (str): Response category (e.g., 'default', 'joy', 'high')
            responses (list): List of response strings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create intent category if it doesn't exist
            if intent not in self.response_repository:
                self.response_repository[intent] = {}
            
            # Create or update response category
            if category in self.response_repository[intent]:
                self.response_repository[intent][category].extend(responses)
            else:
                self.response_repository[intent][category] = responses
            
            return True
        except Exception as e:
            print(f"Error adding responses: {e}")
            return False
    
    def save_repository(self, file_path):
        """
        Save the response repository to a JSON file.
        
        Args:
            file_path (str): Path to save the repository
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.response_repository, f, indent=2)
            print(f"Response repository saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving response repository: {e}")
            return False