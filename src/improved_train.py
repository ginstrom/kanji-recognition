"""
Trains an improved Kanji OCR model using PyTorch, with advanced training techniques.

This module implements an improved CNN model for recognizing handwritten kanji characters
and provides enhanced functionality for training and evaluating the model on the ETL9G dataset.
The improved architecture features residual connections, batch normalization, and dropout
for better performance.
"""
import os
import time
import json
import logging
import argparse
from typing import Tuple, Dict, Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Import functions from load.py
from load import get_dataloaders, get_num_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed_value: int = 42) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Disabling benchmark mode and enabling deterministic mode can impact performance
        # but is good for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np_random_seed = seed_value # for numpy if used by other parts like augmentations
    # import numpy as np; np.random.seed(np_random_seed) # If numpy is used directly for randomness
    logger.info(f"Random seed set to {seed_value}")


class ResidualBlock(nn.Module):
    """
    A single residual block with two convolutional layers, batch normalization,
    ReLU activation, and a skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the residual block."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


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


class ImprovedKanjiCNN(nn.Module):
    """
    An improved CNN model for kanji recognition that incorporates modern best practices
    such as residual connections, batch normalization, and dropout.
    
    The architecture is inspired by ResNet but adapted for grayscale input and
    kanji classification with a focus on recognizing intricate stroke patterns.
    
    Attributes:
        initial_conv (nn.Conv2d): Initial convolutional layer
        bn_initial (nn.BatchNorm2d): Batch normalization after initial convolution
        res_blocks (nn.ModuleList): List of residual blocks
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling
        dropout (nn.Dropout): Dropout layer for regularization
        fc (nn.Linear): Final fully connected layer for classification
    """
    def __init__(self, num_classes=3036, dropout_rate=0.5):
        """
        Initialize the ImprovedKanjiCNN model.
        
        Args:
            num_classes (int, optional): Number of output classes. Defaults to 3036.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.5.
        """
        super(ImprovedKanjiCNN, self).__init__()
        
        # Initial convolution layer
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_initial = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with increasing channels
        self.res_blocks = nn.ModuleList([
            self._make_res_layer(64, 64, blocks=2, stride=1),
            self._make_res_layer(64, 128, blocks=2, stride=2),
            self._make_res_layer(128, 256, blocks=2, stride=2),
            self._make_res_layer(256, 512, blocks=2, stride=2)
        ])
        
        # Global average pooling and final classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_res_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """
        Create a layer of residual blocks.
        
        Args:
            in_channels (int): Number of input channels for the first block.
            out_channels (int): Number of output channels for all blocks in this layer.
            blocks (int): Number of residual blocks in this layer.
            stride (int): Stride for the first block in this layer.
            
        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
        layers = []
        # First block might have a stride and change in channels
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Subsequent blocks in the same layer
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization for better training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution and pooling
        x = self.initial_conv(x)
        x = self.bn_initial(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        for res_layer in self.res_blocks:
            x = res_layer(x)
        
        # Global pooling, dropout, and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def plot_training_history(history: Dict[str, Any], output_dir: str) -> None:
    """
    Plots and saves training and validation loss and accuracy.

    Args:
        history (Dict[str, Any]): Dictionary containing training history
                                  (e.g., train_loss, val_loss, train_acc, val_acc).
        output_dir (str): Directory to save the plots.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(fig_dir, 'loss_history.png')
        try:
            plt.savefig(plot_path)
            logger.info(f"Loss history plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save loss history plot to {plot_path}: {e}")
        plt.close() # Close the figure to free memory

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(fig_dir, 'accuracy_history.png')
        try:
            plt.savefig(plot_path)
            logger.info(f"Accuracy history plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save accuracy history plot to {plot_path}: {e}")
        plt.close() # Close the figure to free memory
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping history plots.")
    except Exception as e:
        logger.error(f"An error occurred during plot generation: {e}")


def train_model_improved(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    device: torch.device = None,
    output_dir: str = '/app/output/models',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4
) -> nn.Module:
    """
    Train the model with improved techniques like learning rate scheduling and gradient clipping.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int, optional): Number of epochs to train for. Defaults to 30.
        device (torch.device, optional): Device to train on. Defaults to None.
        output_dir (str, optional): Directory to save model. Defaults to '/app/output/models'.
        learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 1e-4.
        
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
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # For early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_kanji_cnn.pth')
    
    # For mapping character strings to indices
    label_to_idx = None
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Try to get label_to_idx from the train_dataset
    # Prefer class_to_idx if available (standard for ImageFolder)
    if hasattr(train_loader.dataset, 'class_to_idx') and train_loader.dataset.class_to_idx:
        label_to_idx = train_loader.dataset.class_to_idx
        logger.info(f"Using 'class_to_idx' from dataset with {len(label_to_idx)} classes.")
    elif hasattr(train_loader.dataset, 'label_to_idx') and train_loader.dataset.label_to_idx:
        label_to_idx = train_loader.dataset.label_to_idx
        logger.info(f"Using 'label_to_idx' from dataset with {len(label_to_idx)} classes.")
    elif hasattr(train_loader.dataset, 'classes') and train_loader.dataset.classes:
        # Fallback if 'classes' attribute (list of class names) exists
        label_to_idx = {cls_name: i for i, cls_name in enumerate(train_loader.dataset.classes)}
        logger.info(f"Constructed label_to_idx from dataset.classes with {len(label_to_idx)} classes.")
    else:
        logger.warning("Could not find 'class_to_idx', 'label_to_idx', or 'classes' on train_loader.dataset. "
                       "Label mapping will be based on encountered labels if they are strings, "
                       "or assumed to be pre-encoded if numerical.")
        # If labels are already indices, this is fine. If they are strings and no mapping is found,
        # this implies an issue with the dataset setup or how get_num_classes works.

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Assuming labels are already numerical indices from the DataLoader
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward pass with autocast
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize with scaler
            scaler.scale(loss).backward()
            # Gradient clipping (applied to unscaled gradients)
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1), 
                'accuracy': 100 * correct / total
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_idx, (images, labels) in enumerate(progress_bar):
                # Assuming labels are already numerical indices from the DataLoader
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': val_loss / (batch_idx + 1), 
                    'accuracy': 100 * correct / total
                })
        
        # Calculate validation statistics
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'label_to_idx': label_to_idx
            }
            try:
                torch.save(checkpoint_data, best_model_path)
                logger.info(f"Saved best model to {best_model_path} with validation accuracy: {best_val_acc:.2f}%")
            except Exception as e:
                logger.error(f"Failed to save best model to {best_model_path}: {e}")
            
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save intermediate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            intermediate_path = os.path.join(output_dir, f'kanji_cnn_epoch_{epoch+1}.pth')
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'label_to_idx': label_to_idx
            }
            try:
                torch.save(checkpoint_data, intermediate_path)
                logger.info(f"Saved intermediate model to {intermediate_path}")
            except Exception as e:
                logger.error(f"Failed to save intermediate model to {intermediate_path}: {e}")
    
    # Calculate training time
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # Load best model
    best_model_checkpoint = None
    if os.path.exists(best_model_path):
        try:
            best_model_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_model_checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {best_model_checkpoint['epoch']} with validation accuracy: {best_model_checkpoint['val_acc']:.2f}%")
        except Exception as e:
            logger.error(f"Failed to load best model from {best_model_path}: {e}")
            best_model_checkpoint = None # Ensure it's None if loading failed
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using the model from the last epoch.")

    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_model_checkpoint['epoch'] if best_model_checkpoint else num_epochs 
    }
    history_path = os.path.join(output_dir, 'training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save training history to {history_path}: {e}")
    
    # Plot training history
    plot_training_history(history, output_dir)

    # Clean up memory
    del best_model_checkpoint
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache after training.")
    
    return model


def main():
    """
    Main function to train the kanji recognition model with the improved CNN architecture.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train improved kanji recognition model')
    parser.add_argument('--data-dir', type=str, default='/app/output/prep',
                        help='Directory containing the LMDB databases (default: /app/output/prep)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for DataLoader (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for DataLoader (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for regularization (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for regularization (default: 0.5)')
    parser.add_argument('--model-dir', type=str, default='/app/output/models',
                        help='Directory to save trained models (default: /app/output/models)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed() # Default seed is 42, can be made configurable via args if needed
    
    logger.info("Starting kanji recognition model training with improved architecture")
    logger.info(f"Configuration: {args}") # Log hyperparameters
    
    # Load data using existing functions from load.py
    logger.info("Loading datasets...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=True,  # Enable augmentation
        num_workers=args.num_workers
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
    logger.info("Creating improved model...")
    
    # Always use the improved model in this version
    model = ImprovedKanjiCNN(
        num_classes=num_classes,
        dropout_rate=args.dropout
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Detect if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Train model with improved training function
    logger.info("Starting training with improved techniques...")
    
    train_model_improved(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        output_dir=args.model_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    logger.info("Training completed successfully!")
    
    # Test the model if test dataset is available
    if 'test' in dataloaders:
        logger.info("Evaluating model on test dataset...")
        test_loader, test_dataset = dataloaders['test']
        
        # Model should already be loaded with best weights if training occurred.
        # If main is run only for testing, model needs to be loaded.
        # The current flow assumes training happened before testing.
        
        # Get the label mapping from the loaded model's checkpoint (if available and needed)
        # For test evaluation, we need the label_to_idx that the model was trained with.
        # This should ideally come from the checkpoint of the loaded model.
        
        label_to_idx_for_test = None
        best_model_path = os.path.join(args.model_dir, 'best_kanji_cnn.pth') # Re-define for clarity
        if os.path.exists(best_model_path):
            try:
                # Re-load checkpoint just to get label_to_idx if not already available
                # or to ensure consistency if model wasn't just trained.
                checkpoint = torch.load(best_model_path, map_location=device)
                # model.load_state_dict(checkpoint['model_state_dict']) # Model is already loaded
                label_to_idx_for_test = checkpoint.get('label_to_idx')
                if label_to_idx_for_test:
                    logger.info("Using label_to_idx from loaded best model checkpoint for test evaluation.")
                else:
                    logger.warning("label_to_idx not found in best model checkpoint for test set.")
                del checkpoint # Clean up
            except Exception as e:
                logger.error(f"Error loading checkpoint for label_to_idx for test set: {e}")
        else:
            logger.warning("Best model checkpoint not found for retrieving label_to_idx for test set.")

        if label_to_idx_for_test is None and hasattr(train_dataset, 'class_to_idx'): # Fallback to train_dataset's map
            label_to_idx_for_test = train_dataset.class_to_idx
            logger.info("Falling back to train_dataset.class_to_idx for test evaluation.")
        elif label_to_idx_for_test is None:
             logger.warning("No label_to_idx available for test set. Evaluation might be incorrect if labels are not already indices.")


        # Evaluate on test set
        model.eval() # Ensure model is in eval mode
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
            
        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(test_loader, desc="Testing"):
                # Assuming labels from test_loader are already numerical indices.
                # If string labels were possible and label_to_idx_for_test was valid, conversion would be:
                # if isinstance(labels[0], str) and label_to_idx_for_test:
                #     label_indices = torch.tensor([label_to_idx_for_test[lbl] for lbl in labels if lbl in label_to_idx_for_test])
                #     if len(label_indices) != len(labels):
                #         logger.warning("Some test labels not in mapping, skipping them.")
                #         # filter images tensor as well or handle appropriately
                #     labels = label_indices
                # elif isinstance(labels[0], str):
                #    logger.warning(f"Test labels are strings but no valid label_to_idx. Skipping batch or ensure numerical labels.")
                #    continue

                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
            
            test_acc = 100 * correct / total
            logger.info(f"Test accuracy: {test_acc:.2f}%")
            
            # Save test results
            test_results_path = os.path.join(args.model_dir, 'test_results.json')
            try:
                with open(test_results_path, 'w') as f:
                    json.dump({'test_accuracy': test_acc}, f, indent=4)
                logger.info(f"Test results saved to {test_results_path}")
            except Exception as e:
                logger.error(f"Failed to save test results to {test_results_path}: {e}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache at the end of main.")

    logger.info("All tasks completed!")


if __name__ == "__main__":
    main()
