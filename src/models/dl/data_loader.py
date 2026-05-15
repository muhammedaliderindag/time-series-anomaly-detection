import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Union

class TimeSeriesWindowDataset(Dataset):
    """
    PyTorch Dataset for Time Series data using a sliding window approach.
    Expects to yield 3D tensors: (sequence_length, features) for each sample,
    which DataLoader will batch into (batch_size, sequence_length, features).
    """
    def __init__(self, data: Union[np.ndarray, list], sequence_length: int):
        """
        Args:
            data (np.ndarray or list): Time series data of shape (num_samples,) or (num_samples, features).
            sequence_length (int): Length of the sliding window.
        """
        data_np = np.array(data)
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
            
        self.data = torch.tensor(data_np, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.num_samples = len(data_np) - sequence_length + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Data length ({len(data_np)}) must be greater than or equal to sequence_length ({sequence_length}).")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single window of data.
        Returns:
            torch.Tensor: Shape (sequence_length, features)
        """
        window = self.data[idx : idx + self.sequence_length]
        return window


def create_dataloader(data: Union[np.ndarray, list], sequence_length: int, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the time series window dataset.
    
    Args:
        data (np.ndarray or list): Time series data.
        sequence_length (int): Length of the sliding window.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset. Default is False.
        num_workers (int): Number of worker threads for data loading. Default is 0.
        
    Returns:
        DataLoader: PyTorch DataLoader yielding batches of shape (batch_size, sequence_length, features)
    """
    dataset = TimeSeriesWindowDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
