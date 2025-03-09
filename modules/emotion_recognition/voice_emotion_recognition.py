import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for speech emotion recognition.
    """
    def __init__(self, input_size=13, hidden_size=64, num_layers=2, num_classes=7):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, input_size)
        
        # Transpose for CNN
        x = x.transpose(1, 2)  # (batch_size, input_size, time_steps)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)  # (batch_size, time_steps, channels)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # (batch_size, time_steps, hidden_size*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Fully connected layer
        output = self.fc(context_vector)
        
        return output

class VoiceEmotionRecognizer:
    """
    A class for recognizing emotions from voice using a CNN-LSTM model.
    """
    
    def __init__(self, model_path=None, input_size=13, hidden_size=64, num_layers=2, num_classes=7, device=None):
        """
        Initialize the VoiceEmotionRecognizer with a pre-trained or new model.
        
        Args:
            model_path (str): Path to a pre-trained model file, or None to create a new model
            input_size (int): Input feature dimension (e.g., number of MFCC features)
            hidden_size (int): Hidden size for LSTM layers
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of emotion classes
            device (str): Device to use for inference ('cuda' or 'cpu'), or None to auto-detect
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Define emotion labels
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.model = CNNLSTM(input_size, hidden_size, num_layers, num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # Create new model
            self.model = CNNLSTM(input_size, hidden_size, num_layers, num_classes)
            print("Created new model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_features(self, features):
        """
        Preprocess audio features for model input.
        
        Args:
            features (numpy.ndarray): Audio features (e.g., MFCCs)
            
        Returns:
            torch.Tensor: Preprocessed features
        """
        # Transpose features if needed (time_steps, features) -> (features, time_steps)
        if features.shape[0] < features.shape[1]:
            features = features.T
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return features_tensor
    
    def predict_emotion(self, features):
        """
        Predict emotions from audio features.
        
        Args:
            features (numpy.ndarray): Audio features (e.g., MFCCs)
            
        Returns:
            dict: Dictionary mapping emotion labels to probabilities
        """
        try:
            # Preprocess features
            features_tensor = self.preprocess_features(features)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Map probabilities to emotion labels
            emotions = {self.emotion_labels[i]: float(probabilities[i]) for i in range(len(self.emotion_labels))}
            return emotions
        except Exception as e:
            print(f"Error predicting emotion from voice: {e}")
            return None
    
    def get_dominant_emotion(self, features):
        """
        Get the dominant emotion from audio features.
        
        Args:
            features (numpy.ndarray): Audio features (e.g., MFCCs)
            
        Returns:
            tuple: (dominant_emotion, probability)
        """
        emotions = self.predict_emotion(features)
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            return dominant_emotion
        return None
    
    def train(self, train_features, train_labels, val_features=None, val_labels=None, 
              epochs=50, batch_size=32, learning_rate=0.001, save_path=None):
        """
        Train the model on a dataset.
        
        Args:
            train_features (list): List of training features (numpy arrays)
            train_labels (list): List of training labels (integers)
            val_features (list): List of validation features (numpy arrays)
            val_labels (list): List of validation labels (integers)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            save_path (str): Path to save the trained model
            
        Returns:
            dict: Training history
        """
        try:
            # Set model to training mode
            self.model.train()
            
            # Prepare data
            train_features = [torch.FloatTensor(f) for f in train_features]
            train_labels = torch.LongTensor(train_labels)
            
            # Pad sequences to the same length
            max_length = max(f.shape[0] for f in train_features)
            train_features_padded = torch.zeros(len(train_features), max_length, train_features[0].shape[1])
            for i, f in enumerate(train_features):
                train_features_padded[i, :f.shape[0], :] = f
            
            # Create dataset and dataloader
            train_dataset = TensorDataset(train_features_padded, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare validation data if provided
            if val_features and val_labels:
                val_features = [torch.FloatTensor(f) for f in val_features]
                val_labels = torch.LongTensor(val_labels)
                
                val_features_padded = torch.zeros(len(val_features), max_length, val_features[0].shape[1])
                for i, f in enumerate(val_features):
                    val_features_padded[i, :f.shape[0], :] = f
                
                val_dataset = TensorDataset(val_features_padded, val_labels)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            else:
                val_loader = None
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = 100 * correct / total
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Validation
                if val_loader:
                    self.model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for features, labels in val_loader:
                            features, labels = features.to(self.device), labels.to(self.device)
                            
                            outputs = self.model(features)
                            loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    val_acc = 100 * correct / total
                    history['val_loss'].append(avg_val_loss)
                    history['val_acc'].append(val_acc)
                    
                    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save the trained model if a path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            
            # Set model back to evaluation mode
            self.model.eval()
            
            return history
        except Exception as e:
            print(f"Error during training: {e}")
            return None