import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Evaluates the model on the given dataloader and returns anomaly scores (reconstruction errors).
    
    Args:
        model (nn.Module): The trained PyTorch autoencoder model.
        dataloader (DataLoader): DataLoader for the data to be evaluated.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        
    Returns:
        np.ndarray: 1D array of anomaly scores (MSE per sequence) of length (num_samples).
    """
    model = model.to(device)
    model.eval()
    
    anomaly_scores = []
    
    # We use reduction='none' to get the error per element
    criterion = nn.MSELoss(reduction='none') 
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Compute MSE per element
            # loss shape: (batch_size, sequence_length, features)
            loss = criterion(outputs, batch)
            
            # Mean over sequence_length and features to get a single score per sequence
            # sample_scores shape: (batch_size,)
            sample_scores = loss.mean(dim=[1, 2]).cpu().numpy()
            
            anomaly_scores.extend(sample_scores)
            
    return np.array(anomaly_scores)


def detect_anomalies(anomaly_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detects anomalies based on a threshold.
    
    Args:
        anomaly_scores (np.ndarray): 1D array of anomaly scores.
        threshold (float): Threshold above which a sample is considered anomalous.
        
    Returns:
        np.ndarray: Boolean array where True indicates an anomaly.
    """
    return anomaly_scores > threshold


def calculate_dynamic_threshold(train_anomaly_scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Calculates a dynamic threshold based on the distribution of training anomaly scores.
    
    Args:
        train_anomaly_scores (np.ndarray): Anomaly scores on the training set (assumed mostly normal).
        percentile (float): Percentile to use for the threshold (e.g., 95.0).
        
    Returns:
        float: Calculated threshold.
    """
    return float(np.percentile(train_anomaly_scores, percentile))
