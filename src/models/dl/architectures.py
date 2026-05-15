import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for Time Series Anomaly Detection.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float = 0.2):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of features in the hidden state of the LSTM.
            latent_dim (int): Dimension of the latent space representation.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability.
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.reconstructor = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (batch_size, sequence_length, input_dim)
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, sequence_length, input_dim)
        """
        seq_len = x.size(1)
        
        # Encode
        # enc_out: (batch_size, seq_len, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim)
        _, (hidden, _) = self.encoder_lstm(x)
        
        # Take the last layer's hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        latent = self.encoder_fc(last_hidden)  # (batch_size, latent_dim)
        
        # Decode
        dec_init = self.decoder_fc(latent)  # (batch_size, hidden_dim)
        
        # Repeat dec_init for each time step to form the sequence for the decoder
        dec_seq = dec_init.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        dec_out, _ = self.decoder_lstm(dec_seq)  # (batch_size, seq_len, hidden_dim)
        
        # Reconstruct
        reconstruction = self.reconstructor(dec_out)  # (batch_size, seq_len, input_dim)
        
        return reconstruction


class CNNAutoencoder(nn.Module):
    """
    1D-CNN-based Autoencoder for Time Series Anomaly Detection.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, sequence_length: int, dropout: float = 0.2):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of base channels for convolutions.
            latent_dim (int): Dimension of the latent space representation.
            sequence_length (int): Length of the input sequence.
            dropout (float): Dropout probability.
        """
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate sequence length after two MaxPool1d(2)
        self.enc_seq_len = sequence_length // 4
        
        # We need to ensure sequence length allows 2 poolings (i.e. is >= 4)
        if self.enc_seq_len == 0:
             self.enc_seq_len = 1
             
        self.flatten_dim = hidden_dim * 2 * self.enc_seq_len
        
        self.encoder_fc = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (batch_size, sequence_length, input_dim)
        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, sequence_length, input_dim)
        """
        orig_seq_len = x.size(1)
        
        # PyTorch Conv1d expects (batch_size, channels, length)
        x = x.transpose(1, 2)
        
        # Encode
        enc_out = self.encoder(x)
        enc_out_flat = enc_out.view(enc_out.size(0), -1)
        latent = self.encoder_fc(enc_out_flat)
        
        # Decode
        dec_out_flat = self.decoder_fc(latent)
        dec_out_reshaped = dec_out_flat.view(dec_out_flat.size(0), -1, self.enc_seq_len)
        
        reconstruction = self.decoder(dec_out_reshaped)
        
        # If output length does not exactly match original due to pooling/padding arithmetic, interpolate
        if reconstruction.size(2) != orig_seq_len:
            reconstruction = nn.functional.interpolate(reconstruction, size=orig_seq_len)
            
        # Back to (batch_size, sequence_length, input_dim)
        reconstruction = reconstruction.transpose(1, 2)
        
        return reconstruction
