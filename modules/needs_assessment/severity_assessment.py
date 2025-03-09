import re
import numpy as np

class SeverityAssessor:
    """
    A class for assessing the severity of mental health needs based on user input and detected emotions.
    """
    
    def __init__(self):
        """
        Initialize the SeverityAssessor with predefined keywords and patterns.
        """
        # Define severity levels and corresponding keywords
        self.critical_keywords = [
            'suicide', 'kill myself', 'end my life', 'don\'t want to live', 'better off dead',
            'no reason to live', 'want to die', 'plan to kill', 'hurt myself', 'self-harm',
            'overdose', 'hanging', 'jump', 'gun', 'pills', 'cut myself'
        ]
        
        self.high_keywords = [
            'hopeless', 'worthless', 'can\'t go on', 'giving up', 'no future',
            'severe depression', 'severe anxiety', 'panic attack', 'trauma',
            'self-harm', 'cutting', 'hurting myself', 'hallucination', 'voices',
            'paranoid', 'delusion', 'manic', 'mania', 'psychosis'
        ]
        
        self.medium_keywords = [
            'sad', 'anxious', 'stressed', 'worried', 'upset',
            'lonely', 'isolated', 'struggling', 'difficulty coping',
            'insomnia', 'can\'t sleep', 'no appetite', 'too much sleep',
            'tired all the time', 'no energy', 'no motivation'
        ]
        
        # Define regex patterns for more complex matching
        self.critical_patterns = [
            r'(want|going|plan|thinking about) to (kill|hurt) (myself|me)',
            r'(don\'t|do not) (want|wish) to (live|be alive|exist)',
            r'(better|best) (if I was|if I were) dead',
            r'(no|any) (reason|point) (in|to) living',
            r'(end|ending) (my|this) (life|suffering|pain)',
            r'(can\'t|cannot) (take|handle|deal with) (it|this) (anymore|any longer)'
        ]
        
        self.high_patterns = [
            r'(feel|feeling) (hopeless|worthless|empty|numb)',
            r'(no|zero) (hope|future|prospects)',
            r'(giving|given) up',
            r'(severe|serious|major) (depression|anxiety|mental health issue)',
            r'(constant|persistent|overwhelming) (fear|worry|sadness|grief)'
        ]
        
        self.medium_patterns = [
            r'(feel|feeling) (sad|anxious|stressed|worried|upset)',
            r'(trouble|difficulty|problem with) (sleeping|eating|concentrating)',
            r'(don\'t|do not) (enjoy|like) (things|activities) (anymore|any longer)',
            r'(low|no) (energy|motivation)',
            r'(isolated|alone|lonely)'
        ]
        
        # Compile regex patterns
        self.critical_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.critical_patterns]
        self.high_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.high_patterns]
        self.medium_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.medium_patterns]
        
        # Define emotion severity mappings
        self.emotion_severity = {
            'anger': {'high': 0.7, 'medium': 0.4, 'low': 0.0},
            'disgust': {'high': 0.6, 'medium': 0.3, 'low': 0.0},
            'fear': {'high': 0.8, 'medium': 0.5, 'low': 0.2},
            'joy': {'high': 0.0, 'medium': 0.0, 'low': 0.0},
            'neutral': {'high': 0.0, 'medium': 0.0, 'low': 0.0},
            'sadness': {'high': 0.8, 'medium': 0.5, 'low': 0.2},
            'surprise': {'high': 0.3, 'medium': 0.1, 'low': 0.0}
        }
        
        # Define intent severity mappings
        self.intent_severity = {
            'suicidal': 4.0,  # Critical
            'depression': 3.0,  # High
            'anxiety': 3.0,    # High
            'mood_unhappy': 2.0,  # Medium
            'help': 2.0,       # Medium
            'resources': 2.0,  # Medium
            'coping': 1.5,     # Medium-Low
            'chitchat': 1.0,   # Low
            'greeting': 1.0,   # Low
            'goodbye': 1.0,    # Low
            'thanks': 1.0,     # Low
            'affirm': 1.0,     # Low
            'deny': 1.0        # Low
        }
    
    def assess_text_severity(self, text):
        """
        Assess the severity of mental health needs based on text content.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing severity assessment
        """
        text = text.lower()
        
        # Check for critical keywords and patterns
        if any(keyword in text for keyword in self.critical_keywords) or \
           any(pattern.search(text) for pattern in self.critical_regex):
            severity = 'critical'
            score = 4.0
        # Check for high severity keywords and patterns
        elif any(keyword in text for keyword in self.high_keywords) or \
             any(pattern.search(text) for pattern in self.high_regex):
            severity = 'high'
            score = 3.0
        # Check for medium severity keywords and patterns
        elif any(keyword in text for keyword in self.medium_keywords) or \
             any(pattern.search(text) for pattern in self.medium_regex):
            severity = 'medium'
            score = 2.0
        # Default to low severity
        else:
            severity = 'low'
            score = 1.0
        
        return {
            'severity_level': severity,
            'severity_score': score,
            'requires_immediate_attention': severity == 'critical',
            'source': 'text_content'
        }
    
    def assess_emotion_severity(self, emotions):
        """
        Assess the severity of mental health needs based on detected emotions.
        
        Args:
            emotions (dict): Dictionary mapping emotion labels to probabilities
            
        Returns:
            dict: Dictionary containing severity assessment
        """
        if not emotions:
            return {
                'severity_level': 'unknown',
                'severity_score': 0.0,
                'requires_immediate_attention': False,
                'source': 'emotion'
            }
        
        # Get the dominant emotion and its probability
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion_name = dominant_emotion[0]
        emotion_prob = dominant_emotion[1]
        
        # Determine severity based on emotion and its intensity
        if emotion_name in self.emotion_severity:
            if emotion_prob >= self.emotion_severity[emotion_name]['high']:
                severity = 'high'
                score = 3.0
            elif emotion_prob >= self.emotion_severity[emotion_name]['medium']:
                severity = 'medium'
                score = 2.0
            else:
                severity = 'low'
                score = 1.0
        else:
            severity = 'low'
            score = 1.0
        
        # Special case for sadness and fear with high probability
        if emotion_name in ['sadness', 'fear'] and emotion_prob > 0.8:
            severity = 'high'
            score = 3.0
        
        return {
            'severity_level': severity,
            'severity_score': score,
            'requires_immediate_attention': False,  # Emotions alone don't trigger immediate attention
            'source': 'emotion',
            'dominant_emotion': emotion_name,
            'emotion_probability': emotion_prob
        }
    
    def assess_intent_severity(self, intent_result):
        """
        Assess the severity of mental health needs based on detected intent.
        
        Args:
            intent_result (dict): Intent prediction result
            
        Returns:
            dict: Dictionary containing severity assessment
        """
        intent_name = intent_result['intent']['name']
        confidence = intent_result['intent']['confidence']
        
        # Get severity score for the intent
        score = self.intent_severity.get(intent_name, 1.0)
        
        # Adjust score based on confidence
        adjusted_score = score * confidence
        
        # Determine severity level
        if adjusted_score >= 3.5:
            severity = 'critical'
        elif adjusted_score >= 2.5:
            severity = 'high'
        elif adjusted_score >= 1.5:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'severity_level': severity,
            'severity_score': adjusted_score,
            'requires_immediate_attention': severity == 'critical',
            'source': 'intent',
            'intent': intent_name,
            'confidence': confidence
        }
    
    def combine_assessments(self, text_assessment, emotion_assessment, intent_assessment):
        """
        Combine multiple severity assessments into a final assessment.
        
        Args:
            text_assessment (dict): Text-based severity assessment
            emotion_assessment (dict): Emotion-based severity assessment
            intent_assessment (dict): Intent-based severity assessment
            
        Returns:
            dict: Dictionary containing combined severity assessment
        """
        # Extract scores
        text_score = text_assessment.get('severity_score', 0.0)
        emotion_score = emotion_assessment.get('severity_score', 0.0)
        intent_score = intent_assessment.get('severity_score', 0.0)
        
        # Calculate weighted average score
        # Text content is given the highest weight as it's most reliable for severity assessment
        weights = {'text': 0.5, 'emotion': 0.2, 'intent': 0.3}
        combined_score = (text_score * weights['text'] + 
                          emotion_score * weights['emotion'] + 
                          intent_score * weights['intent'])
        
        # Determine combined severity level
        if combined_score >= 3.5 or text_assessment.get('severity_level') == 'critical':
            severity = 'critical'
        elif combined_score >= 2.5:
            severity = 'high'
        elif combined_score >= 1.5:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Determine if immediate attention is required
        requires_attention = (severity == 'critical' or 
                             text_assessment.get('requires_immediate_attention', False) or
                             intent_assessment.get('requires_immediate_attention', False))
        
        return {
            'severity_level': severity,
            'severity_score': combined_score,
            'requires_immediate_attention': requires_attention,
            'component_assessments': {
                'text': text_assessment,
                'emotion': emotion_assessment,
                'intent': intent_assessment
            }
        }
    
    def assess_severity(self, text, emotions=None, intent_result=None):
        """
        Assess the overall severity of mental health needs.
        
        Args:
            text (str): Input text
            emotions (dict): Dictionary mapping emotion labels to probabilities (optional)
            intent_result (dict): Intent prediction result (optional)
            
        Returns:
            dict: Dictionary containing severity assessment
        """
        # Perform text-based assessment
        text_assessment = self.assess_text_severity(text)
        
        # Perform emotion-based assessment if emotions are provided
        if emotions:
            emotion_assessment = self.assess_emotion_severity(emotions)
        else:
            emotion_assessment = {
                'severity_level': 'unknown',
                'severity_score': 0.0,
                'requires_immediate_attention': False,
                'source': 'emotion'
            }
        
        # Perform intent-based assessment if intent result is provided
        if intent_result:
            intent_assessment = self.assess_intent_severity(intent_result)
        else:
            intent_assessment = {
                'severity_level': 'unknown',
                'severity_score': 0.0,
                'requires_immediate_attention': False,
                'source': 'intent'
            }
        
        # Combine assessments
        combined_assessment = self.combine_assessments(
            text_assessment, emotion_assessment, intent_assessment
        )
        
        return combined_assessment