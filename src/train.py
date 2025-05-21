"""
Kanji Recognition Model Training Module

This module implements a simple CNN model for recognizing handwritten kanji characters
and provides functionality for training and evaluating the model on the ETL9G dataset.
"""
import os
import time
import logging
# from typing import Tuple, Dict, Optional # Unused F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Import functions from load.py
from load import get_dataloaders, get_num_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleKanjiCNN(nn.Module):
    """
    A simple CNN model for kanji recognition with two convolutional layers.
    
    This model is designed as a baseline for kanji recognition, with a simple
    architecture that can be trained quickly to establish a working pipeline.
    
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer
        conv2 (nn.Conv2d): Second convolutional layer
        pool (nn.MaxPool2d): Max pooling layer
        fc (nn.Linear): Fully connected layer for classification
    """
    def __init__(self, num_classes: int = 3036):
        """
        Initialize the SimpleKanjiCNN model.
        
        Args:
            num_classes (int, optional): Number of output classes. Defaults to 3036.
        """
        super(SimpleKanjiCNN, self).__init__()
        # Two convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # One fully connected layer
        self.fc = nn.Linear(64 * 32 * 32, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.fc(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 5,
    device: torch.device = None,
    output_dir: str = '/app/output/models'
) -> nn.Module:
    """
    Train the model on the provided data.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int, optional): Number of epochs to train for. Defaults to 5.
        device (torch.device, optional): Device to train on. Defaults to None.
        output_dir (str, optional): Directory to save model. Defaults to '/app/output/models'.
        
    Returns:
        nn.Module: The trained model
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Training on device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Convert string labels to indices if necessary
            if isinstance(labels[0], str):
                # Create a mapping of characters to indices
                unique_chars = sorted(set(labels))
                char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
                # Convert labels to indices
                label_indices = torch.tensor([char_to_idx[label] for label in labels])
                labels = label_indices
            
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch statistics
            if (i + 1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        # Validation phase (every 2 epochs to save time)
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    # Convert string labels to indices if necessary
                    if isinstance(labels[0], str):
                        # Create a mapping of characters to indices
                        unique_chars = sorted(set(labels))
                        char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
                        # Convert labels to indices
                        label_indices = torch.tensor([char_to_idx[label] for label in labels])
                        labels = label_indices
                    
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            logger.info(f'Validation Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    # Save the final model
    model_path = os.path.join(output_dir, 'simple_kanji_cnn.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')
    
    return model


def main():
    """
    Main function to train the kanji recognition model.
    """
    logger.info("Starting kanji recognition model training")
    
    # Load data using existing functions from load.py
    logger.info("Loading datasets...")
    dataloaders = get_dataloaders(
        data_dir='/app/output/prep',
        batch_size=64,
        augment=False,  # No augmentation for simplicity
        num_workers=4
    )
    
    if 'train' not in dataloaders or 'val' not in dataloaders:
        logger.error("Could not load train or validation datasets")
        return
    
    train_loader, train_dataset = dataloaders['train']
    val_loader, val_dataset = dataloaders['val']
    
    # Get number of classes
    num_classes = get_num_classes(train_dataset)
    logger.info(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleKanjiCNN(num_classes)
    
    # Train model
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=5)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
