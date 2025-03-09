import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms

class EmotionCNN(nn.Module):
    """
    CNN model for facial emotion recognition.
    """
    def __init__(self, num_classes=7, use_pretrained=True):
        super(EmotionCNN, self).__init__()
        
        # Use a pre-trained ResNet model
        if use_pretrained:
            self.base_model = models.resnet18(pretrained=True)
            # Replace the last fully connected layer
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)
        else:
            # Define a custom CNN architecture
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(512 * 3 * 3, 256)
            self.fc2 = nn.Linear(256, num_classes)
            
            self.use_pretrained = use_pretrained
    
    def forward(self, x):
        if hasattr(self, 'use_pretrained') and not self.use_pretrained:
            # Custom CNN forward pass
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
        else:
            # Pre-trained model forward pass
            x = self.base_model(x)
        
        return x

class FacialEmotionRecognizer:
    """
    A class for recognizing emotions from facial expressions using a CNN model.
    """
    
    def __init__(self, model_path=None, num_classes=7, use_pretrained=True, device=None):
        """
        Initialize the FacialEmotionRecognizer with a pre-trained or new model.
        
        Args:
            model_path (str): Path to a pre-trained model file, or None to create a new model
            num_classes (int): Number of emotion classes
            use_pretrained (bool): Whether to use a pre-trained ResNet model
            device (str): Device to use for inference ('cuda' or 'cpu'), or None to auto-detect
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Define emotion labels
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.grayscale_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Create or load model
        if model_path and os.path.exists(model_path):
            # Load pre-trained model
            self.model = EmotionCNN(num_classes, use_pretrained)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # Create new model
            self.model = EmotionCNN(num_classes, use_pretrained)
            print("Created new model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store model configuration
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
    
    def preprocess_image(self, image):
        """
        Preprocess an image for model input.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Check if image is grayscale or RGB
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                # Grayscale image
                if self.use_pretrained:
                    # Convert to 3-channel for pre-trained model
                    image = np.stack((image,) * 3, axis=-1)
                    tensor = self.transform(image)
                else:
                    # Keep as 1-channel for custom model
                    tensor = self.grayscale_transform(image)
            else:
                # RGB image
                if not self.use_pretrained:
                    # Convert to grayscale for custom model
                    tensor = self.grayscale_transform(image)
                else:
                    # Keep as RGB for pre-trained model
                    tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_emotion(self, image):
        """
        Predict emotions from a facial image.
        
        Args:
            image (numpy.ndarray): Input facial image
            
        Returns:
            dict: Dictionary mapping emotion labels to probabilities
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return None
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Map probabilities to emotion labels
            emotions = {self.emotion_labels[i]: float(probabilities[i]) for i in range(len(self.emotion_labels))}
            return emotions
        except Exception as e:
            print(f"Error predicting emotion from facial expression: {e}")
            return None
    
    def get_dominant_emotion(self, image):
        """
        Get the dominant emotion from a facial image.
        
        Args:
            image (numpy.ndarray): Input facial image
            
        Returns:
            tuple: (dominant_emotion, probability)
        """
        emotions = self.predict_emotion(image)
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            return dominant_emotion
        return None
    
    def train(self, train_images, train_labels, val_images=None, val_labels=None, 
              epochs=50, batch_size=32, learning_rate=0.001, save_path=None):
        """
        Train the model on a dataset.
        
        Args:
            train_images (list): List of training images (numpy arrays)
            train_labels (list): List of training labels (integers)
            val_images (list): List of validation images (numpy arrays)
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
            train_tensors = []
            for image in train_images:
                if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                    # Grayscale image
                    if self.use_pretrained:
                        # Convert to 3-channel for pre-trained model
                        image = np.stack((image,) * 3, axis=-1)
                        tensor = self.transform(image)
                    else:
                        # Keep as 1-channel for custom model
                        tensor = self.grayscale_transform(image)
                else:
                    # RGB image
                    if not self.use_pretrained:
                        # Convert to grayscale for custom model
                        tensor = self.grayscale_transform(image)
                    else:
                        # Keep as RGB for pre-trained model
                        tensor = self.transform(image)
                train_tensors.append(tensor)
            
            train_tensors = torch.stack(train_tensors)
            train_labels = torch.LongTensor(train_labels)
            
            # Create dataset and dataloader
            train_dataset = TensorDataset(train_tensors, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare validation data if provided
            if val_images and val_labels:
                val_tensors = []
                for image in val_images:
                    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                        # Grayscale image
                        if self.use_pretrained:
                            # Convert to 3-channel for pre-trained model
                            image = np.stack((image,) * 3, axis=-1)
                            tensor = self.transform(image)
                        else:
                            # Keep as 1-channel for custom model
                            tensor = self.grayscale_transform(image)
                    else:
                        # RGB image
                        if not self.use_pretrained:
                            # Convert to grayscale for custom model
                            tensor = self.grayscale_transform(image)
                        else:
                            # Keep as RGB for pre-trained model
                            tensor = self.transform(image)
                    val_tensors.append(tensor)
                
                val_tensors = torch.stack(val_tensors)
                val_labels = torch.LongTensor(val_labels)
                
                val_dataset = TensorDataset(val_tensors, val_labels)
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
                
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
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
                        for images, labels in val_loader:
                            images, labels = images.to(self.device), labels.to(self.device)
                            
                            outputs = self.model(images)
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