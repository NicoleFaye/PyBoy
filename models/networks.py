"""
Neural network architectures for various agents.
"""
import torch
from torch import nn


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Pokemon Pinball.
    Architecture: CNN with multiple convolutional layers, followed by fully connected layers.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize the network.
        
        Args:
            input_dim: Input dimensions (channels, height, width)
            output_dim: Number of output actions
        """
        super().__init__()
        c, h, w = input_dim
        
        if h != 16:
            raise ValueError(f"Expecting input height: 16, got: {h}")
        if w != 20:
            raise ValueError(f"Expecting input width: 20, got: {w}")
            
        # Online network for action selection
        self.online = self._build_cnn(c, output_dim)
        
        # Target network for stable learning
        self.target = self._build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        
        # Freeze target network parameters
        for p in self.target.parameters():
            p.requires_grad = False
            
    def forward(self, input, model):
        """
        Forward pass through the network.
        
        Args:
            input: Input tensor
            model: Which model to use ('online' or 'target')
            
        Returns:
            Output tensor
        """
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"Model {model} not recognized. Use 'online' or 'target'.")
            
    def _build_cnn(self, in_channels, output_dim):
        """
        Build the CNN architecture.
        
        Args:
            in_channels: Number of input channels
            output_dim: Number of output actions
            
        Returns:
            CNN model
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class SimpleCNN(nn.Module):
    """
    A simpler CNN architecture that can be used for supervised learning or as a feature extractor.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize the network.
        
        Args:
            input_dim: Input dimensions (channels, height, width)
            output_dim: Number of output classes or features
        """
        super().__init__()
        c, h, w = input_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate the size after convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.features(dummy_input)
            feature_size = dummy_output.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x