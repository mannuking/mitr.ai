import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """
    A class for preprocessing text data for NLP tasks.
    """
    
    def __init__(self, remove_stopwords=True, stemming=False, lemmatization=True):
        """
        Initialize the TextPreprocessor with specified options.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            stemming (bool): Whether to apply stemming
            lemmatization (bool): Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        self.stemmer = PorterStemmer() if stemming else None
        self.lemmatizer = WordNetLemmatizer() if lemmatization else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
    
    def preprocess(self, text):
        """
        Preprocess the input text.
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            list: A list of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if enabled
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization if enabled
        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_for_emotion(self, text):
        """
        Preprocess text specifically for emotion recognition tasks.
        Preserves more of the original text features that might be relevant for emotion.
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            str: Preprocessed text ready for emotion recognition
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep emoticons and punctuation that might indicate emotion
        text = re.sub(r'[^a-zA-Z\s\!\?\.\,\:\;\-\_\(\)\[\]\{\}\"\'\<\>\=\+\*\&\^\%\$\#\@\~\`]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_for_intent(self, text):
        """
        Preprocess text specifically for intent recognition tasks.
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            str: Preprocessed text ready for intent recognition
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep question marks and other relevant punctuation
        text = re.sub(r'[^a-zA-Z\s\!\?\.\,]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text