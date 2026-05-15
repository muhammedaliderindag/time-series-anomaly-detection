import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm

from .early_stopping import EarlyStopping

class ModelTrainer:
    """
    Core training loop for Deep Learning models.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            config (Dict): Configuration dictionary containing training hyperparameters.
            device (torch.device): Device to run training on (CPU or GPU).
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.max_epoch = self.config.get('max_epoch', 50)
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.patience = self.config.get('early_stopping', {}).get('patience', 5)
        
        # Loss and Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        model_dir = self.config.get('paths', {}).get('model_dir', './models')
        self.checkpoint_path = f"{model_dir}/best_dl_model.pt"
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.checkpoint_path)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[list, list]:
        """
        Executes the training loop.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            
        Returns:
            Tuple[list, list]: Training and validation loss histories.
        """
        train_losses = []
        val_losses = []

        for epoch in range(1, self.max_epoch + 1):
            # Training Phase
            self.model.train()
            train_loss_epoch = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epoch} [Train]")
            for batch in pbar:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss_epoch += loss.item() * batch.size(0)
                pbar.set_postfix({'loss': loss.item()})
                
            train_loss_epoch /= len(train_loader.dataset)
            train_losses.append(train_loss_epoch)
            
            # Validation Phase
            self.model.eval()
            val_loss_epoch = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                    val_loss_epoch += loss.item() * batch.size(0)
            
            val_loss_epoch /= len(val_loader.dataset)
            val_losses.append(val_loss_epoch)
            
            print(f"Epoch {epoch}/{self.max_epoch} - Train Loss: {train_loss_epoch:.6f} - Val Loss: {val_loss_epoch:.6f}")
            
            # Early Stopping check
            self.early_stopping(val_loss_epoch, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Training stopped.")
                break
                
        # Load the best model weights
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Loaded best model weights from checkpoint.")
        
        return train_losses, val_losses
