import os
import json
import numpy as np
import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

logger = logging.getLogger('chatbot_utils')

def setup_directories(directories):
    """
    Create directories if they don't exist.
    
    Args:
        directories (list): List of directory paths to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created or already exists: {directory}")
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded data, or None if an error occurred
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Data loaded from {file_path}")
            return data
        else:
            logger.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return None

def save_json(data, file_path):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str): Path to the JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")
        return False

def generate_session_id():
    """
    Generate a unique session ID.
    
    Returns:
        str: Session ID
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = np.random.randint(1000, 9999)
    return f"session_{timestamp}_{random_suffix}"

def get_time_of_day():
    """
    Get the time of day (morning, afternoon, evening, night).
    
    Returns:
        str: Time of day
    """
    hour = datetime.datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

def anonymize_data(data, fields_to_anonymize):
    """
    Anonymize sensitive data.
    
    Args:
        data (dict): Data to anonymize
        fields_to_anonymize (list): List of field names to anonymize
        
    Returns:
        dict: Anonymized data
    """
    if not isinstance(data, dict):
        return data
    
    anonymized_data = data.copy()
    
    for field in fields_to_anonymize:
        if field in anonymized_data:
            if isinstance(anonymized_data[field], str):
                anonymized_data[field] = "ANONYMIZED"
            elif isinstance(anonymized_data[field], dict):
                anonymized_data[field] = anonymize_data(anonymized_data[field], fields_to_anonymize)
            elif isinstance(anonymized_data[field], list):
                anonymized_data[field] = [anonymize_data(item, fields_to_anonymize) if isinstance(item, (dict, list)) else "ANONYMIZED" if isinstance(item, str) else item for item in anonymized_data[field]]
    
    return anonymized_data

def log_conversation(conversation_data, log_dir, anonymize=True):
    """
    Log conversation data to a file.
    
    Args:
        conversation_data (dict): Conversation data to log
        log_dir (str): Directory to save the log file
        anonymize (bool): Whether to anonymize sensitive data
        
    Returns:
        str: Path to the log file, or None if an error occurred
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log file name
        session_id = conversation_data.get('session_id', generate_session_id())
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"conversation_{session_id}_{timestamp}.json")
        
        # Anonymize data if requested
        if anonymize:
            fields_to_anonymize = ['user_id', 'user_name', 'email', 'phone', 'address']
            conversation_data = anonymize_data(conversation_data, fields_to_anonymize)
        
        # Save data to file
        save_json(conversation_data, log_file)
        
        logger.info(f"Conversation logged to {log_file}")
        return log_file
    except Exception as e:
        logger.error(f"Error logging conversation: {e}")
        return None

def extract_keywords(text, num_keywords=5):
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text (str): Input text
        num_keywords (int): Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    try:
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                     'from', 'of', 'that', 'this', 'these', 'those', 'it', 'i', 'you', 
                     'he', 'she', 'they', 'we', 'me', 'him', 'her', 'them', 'us', 'my', 
                     'your', 'his', 'their', 'our', 'be', 'been', 'being', 'have', 'has', 
                     'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 
                     'can', 'could', 'may', 'might', 'must', 'if', 'then', 'else', 'when', 
                     'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                     'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                     'so', 'than', 'too', 'very'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:num_keywords]]
        
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def calculate_sentiment_score(emotions):
    """
    Calculate a sentiment score from emotion probabilities.
    
    Args:
        emotions (dict): Dictionary mapping emotion labels to probabilities
        
    Returns:
        float: Sentiment score (-1 to 1, where -1 is negative and 1 is positive)
    """
    if not emotions:
        return 0.0
    
    # Define emotion valence (positive or negative)
    emotion_valence = {
        'anger': -0.8,
        'disgust': -0.6,
        'fear': -0.7,
        'joy': 0.9,
        'neutral': 0.0,
        'sadness': -0.9,
        'surprise': 0.3
    }
    
    # Calculate weighted average of emotion valences
    total_score = 0.0
    total_weight = 0.0
    
    for emotion, probability in emotions.items():
        if emotion in emotion_valence:
            total_score += emotion_valence[emotion] * probability
            total_weight += probability
    
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0

def get_conversation_metrics(conversation_data):
    """
    Calculate metrics from conversation data.
    
    Args:
        conversation_data (dict): Conversation data
        
    Returns:
        dict: Conversation metrics
    """
    try:
        metrics = {}
        
        # Calculate conversation duration
        if 'start_time' in conversation_data and 'last_activity' in conversation_data:
            start_time = datetime.datetime.fromisoformat(conversation_data['start_time'])
            last_activity = datetime.datetime.fromisoformat(conversation_data['last_activity'])
            duration_seconds = (last_activity - start_time).total_seconds()
            metrics['duration_seconds'] = duration_seconds
        
        # Count turns
        if 'conversation_history' in conversation_data:
            metrics['turns'] = len(conversation_data['conversation_history'])
        
        # Calculate average sentiment
        if 'detected_emotions' in conversation_data:
            sentiment_scores = []
            for item in conversation_data['detected_emotions']:
                if 'fused' in item and item['fused']:
                    sentiment_score = calculate_sentiment_score(item['fused'])
                    sentiment_scores.append(sentiment_score)
            
            if sentiment_scores:
                metrics['average_sentiment'] = sum(sentiment_scores) / len(sentiment_scores)
                metrics['sentiment_trend'] = sentiment_scores[-1] - sentiment_scores[0] if len(sentiment_scores) > 1 else 0
        
        # Calculate intent distribution
        if 'detected_intents' in conversation_data:
            intent_counts = {}
            for item in conversation_data['detected_intents']:
                intent = item.get('intent')
                if intent:
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            metrics['intent_distribution'] = intent_counts
            metrics['most_frequent_intent'] = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None
        
        # Calculate severity metrics
        if 'severity_history' in conversation_data:
            severity_levels = [item.get('level') for item in conversation_data['severity_history'] if 'level' in item]
            severity_scores = [item.get('score') for item in conversation_data['severity_history'] if 'score' in item]
            
            if severity_levels:
                severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                metrics['highest_severity'] = max(severity_levels, key=lambda x: severity_map.get(x, 0))
                metrics['average_severity_score'] = sum(severity_scores) / len(severity_scores) if severity_scores else 0
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating conversation metrics: {e}")
        return {}