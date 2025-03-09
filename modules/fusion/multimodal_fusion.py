import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MultimodalFusionModel(nn.Module):
    """
    Neural network model for multimodal fusion of emotion predictions.
    """
    def __init__(self, input_dims, shared_dim=64, num_classes=7):
        """
        Initialize the MultimodalFusionModel.
        
        Args:
            input_dims (dict): Dictionary mapping modality names to their input dimensions
            shared_dim (int): Dimension of the shared representation
            num_classes (int): Number of emotion classes
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.modalities = list(input_dims.keys())
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, shared_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(shared_dim * 2, shared_dim),
                nn.ReLU()
            ) for modality, dim in input_dims.items()
        })
        
        # Attention mechanism
        self.attention = nn.ModuleDict({
            modality: nn.Linear(shared_dim, 1) for modality in self.modalities
        })
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, num_classes)
        )
    
    def forward(self, inputs):
        """
        Forward pass through the fusion model.
        
        Args:
            inputs (dict): Dictionary mapping modality names to their input tensors
            
        Returns:
            torch.Tensor: Output tensor with emotion predictions
        """
        # Encode each modality
        encoded = {modality: self.encoders[modality](inputs[modality]) for modality in self.modalities if modality in inputs}
        
        if not encoded:
            # No modalities provided
            return None
        
        # Calculate attention weights
        attention_weights = {modality: torch.sigmoid(self.attention[modality](encoded[modality])) for modality in encoded}
        
        # Normalize attention weights
        total_weight = sum(attention_weights.values())
        normalized_weights = {modality: weight / total_weight for modality, weight in attention_weights.items()}
        
        # Apply attention weights
        weighted_encodings = [normalized_weights[modality] * encoded[modality] for modality in encoded]
        
        # Sum weighted encodings
        fused = sum(weighted_encodings)
        
        # Final prediction
        output = self.fusion(fused)
        
        return output

class MultimodalFusion:
    """
    A class for fusing emotion predictions from multiple modalities.
    """
    
    def __init__(self, model_path=None, input_dims=None, shared_dim=64, num_classes=7, device=None):
        """
        Initialize the MultimodalFusion with a pre-trained or new model.
        
        Args:
            model_path (str): Path to a pre-trained model file, or None to create a new model
            input_dims (dict): Dictionary mapping modality names to their input dimensions
            shared_dim (int): Dimension of the shared representation
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
        
        # Set default input dimensions if not provided
        if input_dims is None:
            self.input_dims = {
                'text': 7,  # Number of emotion classes from text model
                'voice': 7,  # Number of emotion classes from voice model
                'face': 7    # Number of emotion classes from facial model
            }
        else:
            self.input_dims = input_dims
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.model = MultimodalFusionModel(self.input_dims, shared_dim, num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained fusion model from {model_path}")
        else:
            # Create new model
            self.model = MultimodalFusionModel(self.input_dims, shared_dim, num_classes)
            print("Created new fusion model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def fuse_emotions(self, emotions_dict):
        """
        Fuse emotion predictions from multiple modalities.
        
        Args:
            emotions_dict (dict): Dictionary mapping modality names to their emotion predictions
                Each prediction should be a dictionary mapping emotion labels to probabilities
            
        Returns:
            dict: Dictionary mapping emotion labels to fused probabilities
        """
        try:
            # Convert emotion dictionaries to tensors
            inputs = {}
            for modality, emotions in emotions_dict.items():
                if emotions:
                    # Create a tensor of emotion probabilities
                    probs = torch.tensor([emotions.get(label, 0.0) for label in self.emotion_labels], 
                                         dtype=torch.float32).to(self.device)
                    inputs[modality] = probs
            
            if not inputs:
                # No valid emotion predictions provided
                return None
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(inputs)
                if outputs is None:
                    return None
                probabilities = F.softmax(outputs, dim=0).cpu().numpy()
            
            # Map probabilities to emotion labels
            fused_emotions = {self.emotion_labels[i]: float(probabilities[i]) for i in range(len(self.emotion_labels))}
            return fused_emotions
        except Exception as e:
            print(f"Error fusing emotions: {e}")
            return None
    
    def get_dominant_emotion(self, emotions_dict):
        """
        Get the dominant emotion from fused predictions.
        
        Args:
            emotions_dict (dict): Dictionary mapping modality names to their emotion predictions
            
        Returns:
            tuple: (dominant_emotion, probability)
        """
        fused_emotions = self.fuse_emotions(emotions_dict)
        if fused_emotions:
            dominant_emotion = max(fused_emotions.items(), key=lambda x: x[1])
            return dominant_emotion
        return None
    
    def rule_based_fusion(self, emotions_dict, weights=None):
        """
        Perform rule-based fusion of emotion predictions as a fallback.
        
        Args:
            emotions_dict (dict): Dictionary mapping modality names to their emotion predictions
            weights (dict): Dictionary mapping modality names to their weights
            
        Returns:
            dict: Dictionary mapping emotion labels to fused probabilities
        """
        if not emotions_dict:
            return None
        
        # Set default weights if not provided
        if weights is None:
            weights = {
                'text': 0.3,
                'voice': 0.3,
                'face': 0.4
            }
        
        # Normalize weights for available modalities
        available_modalities = [m for m in emotions_dict if emotions_dict[m]]
        if not available_modalities:
            return None
        
        available_weights = {m: weights.get(m, 1.0) for m in available_modalities}
        total_weight = sum(available_weights.values())
        normalized_weights = {m: w / total_weight for m, w in available_weights.items()}
        
        # Initialize fused emotions
        fused_emotions = {label: 0.0 for label in self.emotion_labels}
        
        # Weighted sum of emotion probabilities
        for modality, emotions in emotions_dict.items():
            if emotions:
                weight = normalized_weights.get(modality, 0.0)
                for label in self.emotion_labels:
                    fused_emotions[label] += weight * emotions.get(label, 0.0)
        
        return fused_emotions
    
    def train(self, train_data, val_data=None, epochs=50, batch_size=32, learning_rate=0.001, save_path=None):
        """
        Train the fusion model on a dataset.
        
        Args:
            train_data (list): List of training samples, each a tuple of (inputs_dict, label)
            val_data (list): List of validation samples, each a tuple of (inputs_dict, label)
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
                
                # Process in batches
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i+batch_size]
                    
                    # Prepare batch data
                    batch_inputs = {}
                    batch_labels = []
                    
                    for inputs_dict, label in batch:
                        # Add inputs for each modality
                        for modality, emotions in inputs_dict.items():
                            if modality not in batch_inputs:
                                batch_inputs[modality] = []
                            
                            # Convert emotions to tensor
                            probs = torch.tensor([emotions.get(label, 0.0) for label in self.emotion_labels], 
                                                dtype=torch.float32)
                            batch_inputs[modality].append(probs)
                        
                        batch_labels.append(label)
                    
                    # Convert lists to tensors
                    for modality in batch_inputs:
                        batch_inputs[modality] = torch.stack(batch_inputs[modality]).to(self.device)
                    
                    batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                
                avg_train_loss = train_loss * batch_size / len(train_data)
                train_acc = 100 * correct / total
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Validation
                if val_data:
                    self.model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        # Process in batches
                        for i in range(0, len(val_data), batch_size):
                            batch = val_data[i:i+batch_size]
                            
                            # Prepare batch data
                            batch_inputs = {}
                            batch_labels = []
                            
                            for inputs_dict, label in batch:
                                # Add inputs for each modality
                                for modality, emotions in inputs_dict.items():
                                    if modality not in batch_inputs:
                                        batch_inputs[modality] = []
                                    
                                    # Convert emotions to tensor
                                    probs = torch.tensor([emotions.get(label, 0.0) for label in self.emotion_labels], 
                                                        dtype=torch.float32)
                                    batch_inputs[modality].append(probs)
                                
                                batch_labels.append(label)
                            
                            # Convert lists to tensors
                            for modality in batch_inputs:
                                batch_inputs[modality] = torch.stack(batch_inputs[modality]).to(self.device)
                            
                            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                            
                            # Forward pass
                            outputs = self.model(batch_inputs)
                            loss = criterion(outputs, batch_labels)
                            
                            val_loss += loss.item()
                            
                            # Calculate accuracy
                            _, predicted = torch.max(outputs.data, 1)
                            total += batch_labels.size(0)
                            correct += (predicted == batch_labels).sum().item()
                    
                    avg_val_loss = val_loss * batch_size / len(val_data)
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