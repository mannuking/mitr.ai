import os
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

class TextEmotionRecognizer:
    """
    A class for recognizing emotions from text using DistilBERT.
    """
    
    def __init__(self, model_path=None, num_labels=7, device=None):
        """
        Initialize the TextEmotionRecognizer with a pre-trained or fine-tuned model.
        
        Args:
            model_path (str): Path to a fine-tuned model directory, or None to use a pre-trained model
            num_labels (int): Number of emotion labels (default: 7 for basic emotions)
            device (str): Device to use for inference ('cuda' or 'cpu'), or None to auto-detect
        """
        self.num_labels = num_labels
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Define emotion labels
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Load model and tokenizer
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=num_labels
            ).to(self.device)
            print(f"Loaded fine-tuned model from {model_path}")
        else:
            # Load pre-trained model
            try:
                # Try to use a pre-trained emotion model from Hugging Face
                self.nlp = pipeline(
                    "text-classification", 
                    model="bhadresh-savani/distilbert-base-uncased-emotion", 
                    top_k=None
                )
                print("Using pre-trained emotion model from Hugging Face")
                self.using_pipeline = True
            except Exception as e:
                print(f"Error loading pre-trained emotion model: {e}")
                print("Falling back to base DistilBERT model")
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', 
                    num_labels=num_labels
                ).to(self.device)
                self.using_pipeline = False
    
    def predict_emotion(self, text):
        """
        Predict emotions from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary mapping emotion labels to probabilities
        """
        try:
            if hasattr(self, 'using_pipeline') and self.using_pipeline:
                # Use the Hugging Face pipeline
                results = self.nlp(text)
                
                if isinstance(results, list) and len(results) == 1:
                    # Handle case where top_k=None returns a list with one item
                    results = results[0]
                
                # Convert to dictionary of emotion -> probability
                emotions = {item['label']: item['score'] for item in results}
                return emotions
            else:
                # Use the model directly
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                
                # Map probabilities to emotion labels
                emotions = {self.emotion_labels[i]: float(probabilities[i]) for i in range(len(self.emotion_labels))}
                return emotions
        except Exception as e:
            print(f"Error predicting emotion from text: {e}")
            return None
    
    def get_dominant_emotion(self, text):
        """
        Get the dominant emotion from text.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (dominant_emotion, probability)
        """
        emotions = self.predict_emotion(text)
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            return dominant_emotion
        return None
    
    def fine_tune(self, train_texts, train_labels, val_texts=None, val_labels=None, 
                  epochs=3, batch_size=16, learning_rate=5e-5, save_path=None):
        """
        Fine-tune the model on a custom dataset.
        
        Args:
            train_texts (list): List of training texts
            train_labels (list): List of training labels (integers)
            val_texts (list): List of validation texts
            val_labels (list): List of validation labels (integers)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            save_path (str): Path to save the fine-tuned model
            
        Returns:
            dict: Training history
        """
        if hasattr(self, 'using_pipeline') and self.using_pipeline:
            print("Fine-tuning not supported when using the Hugging Face pipeline.")
            print("Please initialize with a base model to enable fine-tuning.")
            return None
        
        try:
            from torch.utils.data import DataLoader, TensorDataset
            from transformers import AdamW
            
            # Tokenize all texts
            train_encodings = self.tokenizer(
                train_texts, 
                truncation=True, 
                padding=True, 
                max_length=128,
                return_tensors="pt"
            )
            
            # Create dataset
            train_input_ids = train_encodings['input_ids']
            train_attention_mask = train_encodings['attention_mask']
            train_labels = torch.tensor(train_labels)
            train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create validation dataset if provided
            if val_texts and val_labels:
                val_encodings = self.tokenizer(
                    val_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=128,
                    return_tensors="pt"
                )
                val_input_ids = val_encodings['input_ids']
                val_attention_mask = val_encodings['attention_mask']
                val_labels = torch.tensor(val_labels)
                val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            else:
                val_loader = None
            
            # Prepare optimizer
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            history = {'train_loss': [], 'val_loss': []}
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = [b.to(self.device) for b in batch]
                    input_ids, attention_mask, labels = batch
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
                
                # Validation
                if val_loader:
                    self.model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = [b.to(self.device) for b in batch]
                            input_ids, attention_mask, labels = batch
                            
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    history['val_loss'].append(avg_val_loss)
                    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")
            
            # Save the fine-tuned model if a path is provided
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Model saved to {save_path}")
            
            return history
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return None