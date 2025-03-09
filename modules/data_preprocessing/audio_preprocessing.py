import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from google.cloud import speech
from google.oauth2 import service_account
import io

class AudioPreprocessor:
    """
    A class for preprocessing audio data for speech emotion recognition and speech-to-text conversion.
    """
    
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512, google_credentials_path=None):
        """
        Initialize the AudioPreprocessor with specified parameters.
        
        Args:
            sample_rate (int): Sample rate for audio processing
            n_mfcc (int): Number of MFCC features to extract
            n_fft (int): FFT window size
            hop_length (int): Hop length for the FFT window
            google_credentials_path (str): Path to Google Cloud credentials JSON file
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.google_credentials_path = google_credentials_path
        
        # Initialize Google Cloud Speech client if credentials are provided
        if google_credentials_path and os.path.exists(google_credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
                self.speech_client = speech.SpeechClient(credentials=credentials)
            except Exception as e:
                print(f"Error initializing Google Speech client: {e}")
                self.speech_client = None
        else:
            self.speech_client = None
            if google_credentials_path:
                print(f"Warning: Google credentials file not found at {google_credentials_path}")
    
    def load_audio(self, file_path):
        """
        Load an audio file using librosa.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
    
    def extract_mfcc(self, audio_data, sample_rate=None):
        """
        Extract MFCC features from audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio time series
            sample_rate (int): Sample rate of the audio data (uses self.sample_rate if None)
            
        Returns:
            numpy.ndarray: MFCC features
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Normalize MFCCs
            mfccs = librosa.util.normalize(mfccs, axis=1)
            
            return mfccs
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None
    
    def extract_features(self, audio_data, sample_rate=None):
        """
        Extract multiple audio features including MFCCs, spectral centroid, and zero crossing rate.
        
        Args:
            audio_data (numpy.ndarray): Audio time series
            sample_rate (int): Sample rate of the audio data (uses self.sample_rate if None)
            
        Returns:
            dict: Dictionary containing extracted features
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            features = {}
            
            # Extract MFCCs
            features['mfcc'] = self.extract_mfcc(audio_data, sample_rate)
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['spectral_centroid'] = spectral_centroid
            
            # Extract zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=audio_data,
                hop_length=self.hop_length
            )
            features['zero_crossing_rate'] = zero_crossing_rate
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data, 
                sr=sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['chroma'] = chroma
            
            return features
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def reduce_noise(self, audio_data):
        """
        Apply simple noise reduction to audio data.
        
        Args:
            audio_data (numpy.ndarray): Audio time series
            
        Returns:
            numpy.ndarray: Noise-reduced audio data
        """
        try:
            # Simple noise reduction by removing low amplitude signals
            # This is a very basic approach - more sophisticated methods could be used
            noise_threshold = 0.005
            audio_data_denoised = audio_data.copy()
            audio_data_denoised[np.abs(audio_data) < noise_threshold] = 0
            return audio_data_denoised
        except Exception as e:
            print(f"Error reducing noise: {e}")
            return audio_data
    
    def audio_to_text(self, audio_file_path=None, audio_data=None, language_code="en-US"):
        """
        Convert audio to text using Google Cloud Speech-to-Text API.
        
        Args:
            audio_file_path (str): Path to the audio file (optional if audio_data is provided)
            audio_data (numpy.ndarray): Audio time series (optional if audio_file_path is provided)
            language_code (str): Language code for speech recognition
            
        Returns:
            str: Transcribed text
        """
        if self.speech_client is None:
            print("Google Speech client not initialized. Cannot perform speech-to-text conversion.")
            return None
        
        try:
            # Prepare audio content
            if audio_file_path:
                with io.open(audio_file_path, "rb") as audio_file:
                    content = audio_file.read()
            elif audio_data is not None:
                # Convert numpy array to bytes
                content = sf.write(
                    io.BytesIO(), 
                    audio_data, 
                    self.sample_rate, 
                    format='WAV'
                ).getvalue()
            else:
                print("Either audio_file_path or audio_data must be provided.")
                return None
            
            # Configure audio
            audio = speech.RecognitionAudio(content=content)
            
            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,
            )
            
            # Perform speech recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Extract transcribed text
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript
            
            return transcript
        except Exception as e:
            print(f"Error in speech-to-text conversion: {e}")
            return None