import os
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import io

class ImagePreprocessor:
    """
    A class for preprocessing image data for facial emotion recognition.
    """
    
    def __init__(self, face_cascade_path=None, target_size=(48, 48), google_credentials_path=None):
        """
        Initialize the ImagePreprocessor with specified parameters.
        
        Args:
            face_cascade_path (str): Path to the Haar cascade XML file for face detection
            target_size (tuple): Target size for face images (width, height)
            google_credentials_path (str): Path to Google Cloud credentials JSON file
        """
        self.target_size = target_size
        
        # Load face cascade if path is provided, otherwise use default
        if face_cascade_path and os.path.exists(face_cascade_path):
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            # Use the default OpenCV Haar cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize Google Cloud Vision client if credentials are provided
        self.google_credentials_path = google_credentials_path
        if google_credentials_path and os.path.exists(google_credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            except Exception as e:
                print(f"Error initializing Google Vision client: {e}")
                self.vision_client = None
        else:
            self.vision_client = None
            if google_credentials_path:
                print(f"Warning: Google credentials file not found at {google_credentials_path}")
    
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV's Haar cascade.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of detected face regions (x, y, w, h)
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def extract_face(self, image, face_region):
        """
        Extract a face from an image based on the detected region.
        
        Args:
            image (numpy.ndarray): Input image
            face_region (tuple): Face region (x, y, w, h)
            
        Returns:
            numpy.ndarray: Extracted face image
        """
        try:
            x, y, w, h = face_region
            face_img = image[y:y+h, x:x+w]
            return face_img
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None
    
    def preprocess_face(self, face_img):
        """
        Preprocess a face image for emotion recognition.
        
        Args:
            face_img (numpy.ndarray): Face image
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        try:
            # Resize to target size
            face_img = cv2.resize(face_img, self.target_size)
            
            # Convert to grayscale
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Normalize pixel values to [0, 1]
            face_img = face_img / 255.0
            
            return face_img
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def detect_emotion_with_google_vision(self, image_path=None, image=None):
        """
        Detect facial emotions using Google Cloud Vision API.
        
        Args:
            image_path (str): Path to the image file (optional if image is provided)
            image (numpy.ndarray): Input image (optional if image_path is provided)
            
        Returns:
            dict: Dictionary containing detected emotions and their likelihood
        """
        if self.vision_client is None:
            print("Google Vision client not initialized. Cannot perform emotion detection.")
            return None
        
        try:
            # Prepare image content
            if image_path:
                with io.open(image_path, 'rb') as image_file:
                    content = image_file.read()
            elif image is not None:
                # Convert numpy array to bytes
                success, encoded_image = cv2.imencode('.jpg', image)
                if not success:
                    print("Error encoding image")
                    return None
                content = encoded_image.tobytes()
            else:
                print("Either image_path or image must be provided.")
                return None
            
            # Create image object
            vision_image = vision.Image(content=content)
            
            # Perform face detection
            response = self.vision_client.face_detection(image=vision_image)
            faces = response.face_annotations
            
            if not faces:
                print("No faces detected in the image.")
                return None
            
            # Extract emotion information from the first face
            face = faces[0]
            emotions = {
                'joy': face.joy_likelihood,
                'sorrow': face.sorrow_likelihood,
                'anger': face.anger_likelihood,
                'surprise': face.surprise_likelihood,
                'under_exposed': face.under_exposed_likelihood,
                'blurred': face.blurred_likelihood,
                'headwear': face.headwear_likelihood
            }
            
            # Convert likelihood enum to string
            likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
            emotions = {emotion: likelihood_name[likelihood] for emotion, likelihood in emotions.items()}
            
            return emotions
        except Exception as e:
            print(f"Error in Google Vision emotion detection: {e}")
            return None
    
    def capture_webcam_frame(self, camera_index=0):
        """
        Capture a single frame from the webcam.
        
        Args:
            camera_index (int): Index of the camera to use
            
        Returns:
            numpy.ndarray: Captured frame
        """
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(camera_index)
            
            # Check if webcam is opened successfully
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return None
            
            # Capture a single frame
            ret, frame = cap.read()
            
            # Release the webcam
            cap.release()
            
            if not ret:
                print("Error: Could not capture frame from webcam")
                return None
            
            return frame
        except Exception as e:
            print(f"Error capturing webcam frame: {e}")
            return None
    
    def process_webcam_for_emotion(self, camera_index=0, use_google_vision=False):
        """
        Capture a frame from webcam and process it for emotion recognition.
        
        Args:
            camera_index (int): Index of the camera to use
            use_google_vision (bool): Whether to use Google Vision API for emotion detection
            
        Returns:
            tuple: (processed_face, emotions) if successful, (None, None) otherwise
        """
        # Capture frame from webcam
        frame = self.capture_webcam_frame(camera_index)
        if frame is None:
            return None, None
        
        if use_google_vision and self.vision_client is not None:
            # Use Google Vision API for emotion detection
            emotions = self.detect_emotion_with_google_vision(image=frame)
            return frame, emotions
        else:
            # Use OpenCV for face detection and preprocessing
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                print("No faces detected in the webcam frame")
                return frame, None
            
            # Process the first detected face
            face_region = faces[0]
            face_img = self.extract_face(frame, face_region)
            if face_img is None:
                return frame, None
            
            processed_face = self.preprocess_face(face_img)
            return processed_face, None  # No emotions detected, just preprocessed face