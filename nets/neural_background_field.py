\
\
\
\
\
\
\
\

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .darknet import BaseConv

class PositionalEncoding(nn.Module):

    def __init__(self, input_dim=3, num_freqs=6):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

        freq_bands = 2.0 ** torch.linspace(0.0, num_freqs-1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)

        self.output_dim = input_dim * (1 + 2 * num_freqs)

    def forward(self, coords):
\
\
\
\
\

        B, N, _ = coords.shape
        encoded = [coords]

        for freq in self.freq_bands:

            encoded.append(torch.sin(freq * np.pi * coords))
            encoded.append(torch.cos(freq * np.pi * coords))

        return torch.cat(encoded, dim=-1)

class SceneEncoder(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5, latent_dim=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        self.temporal_aggregator = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels//2, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels//2),
            nn.ReLU(),
            nn.Conv3d(feature_channels//2, feature_channels//4, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels//4),
            nn.ReLU(),
        )

        self.scene_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
            nn.Linear(feature_channels//4 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

        self.dynamics_encoder = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels//2, (2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(feature_channels//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 2, 2)),
            nn.Flatten(),
            nn.Linear(feature_channels//2 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim//2),
            nn.Tanh()
        )

    def forward(self, frame_features):
\
\
\
\
\
\

        feature_volume = torch.stack(frame_features, dim=2)

        aggregated = self.temporal_aggregator(feature_volume)

        scene_code = self.scene_encoder(aggregated)

        dynamics_code = self.dynamics_encoder(feature_volume)

        full_code = torch.cat([scene_code, dynamics_code], dim=1)

        return full_code

class BackgroundNeuralField(nn.Module):

    def __init__(self, scene_code_dim=96, pos_encode_dim=21, feature_dim=128, hidden_dim=256, num_layers=6):
        super().__init__()
        self.scene_code_dim = scene_code_dim
        self.pos_encode_dim = pos_encode_dim
        self.feature_dim = feature_dim

        input_dim = scene_code_dim + pos_encode_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, feature_dim)

        self.activation = nn.ReLU()

        if num_layers > 4:
            self.skip_layer = len(self.hidden_layers) // 2
            self.skip_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        else:
            self.skip_layer = None

    def forward(self, coords, scene_code):
\
\
\
\
\
\

        B, N, _ = coords.shape

        scene_code_expanded = scene_code.unsqueeze(1).expand(-1, N, -1)

        x = torch.cat([coords, scene_code_expanded], dim=-1)
        input_x = x

        x = self.activation(self.input_layer(x))

        for i, layer in enumerate(self.hidden_layers):
            if self.skip_layer is not None and i == self.skip_layer:

                x = torch.cat([x, input_x], dim=-1)
                x = self.activation(self.skip_proj(x))
            else:
                x = self.activation(layer(x))

        features = self.output_layer(x)

        return features

class NeuralBackgroundField(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5, pos_freq=6):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames

        self.pos_encoder = PositionalEncoding(input_dim=3, num_freqs=pos_freq)
        self.scene_encoder = SceneEncoder(
            feature_channels=feature_channels,
            num_frames=num_frames,
            latent_dim=64
        )
        self.neural_field = BackgroundNeuralField(
            scene_code_dim=64 + 32,
            pos_encode_dim=self.pos_encoder.output_dim,
            feature_dim=feature_channels
        )

        self.register_parameter('time_scale', nn.Parameter(torch.tensor(1.0)))

    def create_coords_grid(self, B, H, W, device, current_time=1.0):

        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        t_grid = torch.full_like(x_grid, current_time * self.time_scale.item())

        coords = torch.stack([x_grid, y_grid, t_grid], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        coords = coords.reshape(B, -1, 3)

        return coords

    def forward(self, frame_features, return_residual=True):
\
\
\
\
\
\

        current_features = frame_features[-1]
        B, C, H, W = current_features.shape
        device = current_features.device

        scene_code = self.scene_encoder(frame_features)

        current_time = 1.0
        coords = self.create_coords_grid(B, H, W, device, current_time)

        encoded_coords = self.pos_encoder(coords)

        predicted_features = self.neural_field(encoded_coords, scene_code)
        predicted_features = predicted_features.reshape(B, H, W, C).permute(0, 3, 1, 2)

        results = {
            'predicted_bg': predicted_features,
            'scene_code': scene_code
        }

        if return_residual:

            residual = torch.norm(current_features - predicted_features, p=2, dim=1, keepdim=True)

            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)

            results.update({
                'reconstruction_residual': residual_norm,
                'neural_field_anomaly': residual_norm
            })

        return results
