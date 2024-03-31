import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PokemonPinballFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(PokemonPinballFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        # Assuming the matrix is the main component and you want to extract features using CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # For the scalar values, you can use a simple fully connected layer
        self.scalar_processor = nn.Sequential(
            nn.Linear(2, 64), # Assuming 2 scalar values: ball_x and ball_y
            nn.ReLU(),
        )

        # Final layer to combine CNN and scalar features
        self.final_layer = nn.Sequential(
            nn.Linear(32*4*5 + 64, features_dim), # Adjust the size according to the output of cnn
            nn.ReLU(),
        )

    def forward(self, observations):
        game_area_obs = observations["game_area"].float().unsqueeze(1) # Add channel dimension
        scalar_obs = torch.cat([observations["ball_x"], observations["ball_y"]], dim=1).float()

        game_area_features = self.cnn(matrix_obs)
        game_area_features = matrix_features.view(matrix_features.size(0), -1)

        scalar_features = self.scalar_processor(scalar_obs)

        combined_features = torch.cat([game_area_features, scalar_features], dim=1)

        return self.final_layer(combined_features)
