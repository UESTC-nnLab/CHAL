\
\
\

import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv

class TemporalMedianBG(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames

        self.smooth = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )

    def forward(self, frame_features, return_residual=True):
\
\
\
\
\

        stacked_features = torch.stack(frame_features, dim=2)

        predicted_bg, _ = torch.median(stacked_features, dim=2)

        predicted_bg = self.smooth(predicted_bg)

        results = {'predicted_bg': predicted_bg}

        if return_residual:
            current_features = frame_features[-1]
            residual = torch.norm(current_features - predicted_bg, p=2, dim=1, keepdim=True)
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            results['reconstruction_residual'] = residual_norm

        return results

class GaussianMixtureBG(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5, num_gaussians=3):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames
        self.num_gaussians = num_gaussians

        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(),
            nn.Conv3d(feature_channels, feature_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU()
        )

        self.gmm_predictor = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels // 2, (num_frames, 1, 1)),
            nn.BatchNorm3d(feature_channels // 2),
            nn.ReLU(),
            nn.Conv3d(feature_channels // 2, num_gaussians * 3, (1, 1, 1))
        )

        self.bg_reconstructor = nn.Sequential(
            nn.Conv2d(num_gaussians * 3, feature_channels, 3, 1, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        )

    def forward(self, frame_features, return_residual=True):
\
\
\

        stacked_features = torch.stack(frame_features, dim=2)

        encoded = self.temporal_encoder(stacked_features)

        gmm_params = self.gmm_predictor(encoded).squeeze(2)

        predicted_bg = self.bg_reconstructor(gmm_params)

        results = {'predicted_bg': predicted_bg, 'gmm_params': gmm_params}

        if return_residual:
            current_features = frame_features[-1]
            residual = torch.norm(current_features - predicted_bg, p=2, dim=1, keepdim=True)
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            results['reconstruction_residual'] = residual_norm

        return results

class LSTMBackgroundBG(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5, hidden_dim=256):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        self.feature_proj = nn.Conv2d(feature_channels, hidden_dim // 4, 1)

        self.lstm = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.bg_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, feature_channels, 3, 1, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        )

    def forward(self, frame_features, return_residual=True):
\
\
\

        B, C, H, W = frame_features[0].shape

        projected_features = []
        for feat in frame_features:
            proj = self.feature_proj(feat)
            projected_features.append(proj)

        seq_features = torch.stack(projected_features, dim=1)

        seq_features = seq_features.permute(0, 3, 4, 1, 2)
        seq_features = seq_features.reshape(B * H * W, self.num_frames, -1)

        lstm_out, _ = self.lstm(seq_features)

        last_hidden = lstm_out[:, -1, :]

        last_hidden = last_hidden.reshape(B, H, W, self.hidden_dim)
        last_hidden = last_hidden.permute(0, 3, 1, 2)

        predicted_bg = self.bg_predictor(last_hidden)

        results = {'predicted_bg': predicted_bg}

        if return_residual:
            current_features = frame_features[-1]
            residual = torch.norm(current_features - predicted_bg, p=2, dim=1, keepdim=True)
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            results['reconstruction_residual'] = residual_norm

        return results

class CNN3DBackgroundBG(nn.Module):

    def __init__(self, feature_channels=128, num_frames=5):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_frames = num_frames

        self.encoder = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(),
            nn.Conv3d(feature_channels, feature_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels * 2),
            nn.ReLU(),
            nn.Conv3d(feature_channels * 2, feature_channels * 2, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_channels * 2),
            nn.ReLU()
        )

        self.temporal_aggregator = nn.Sequential(
            nn.Conv3d(feature_channels * 2, feature_channels, (num_frames, 1, 1)),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU()
        )

        self.refiner = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, 1, 1)
        )

    def forward(self, frame_features, return_residual=True):
\
\
\

        stacked_features = torch.stack(frame_features, dim=2)

        encoded = self.encoder(stacked_features)

        aggregated = self.temporal_aggregator(encoded).squeeze(2)

        predicted_bg = self.refiner(aggregated)

        results = {'predicted_bg': predicted_bg}

        if return_residual:
            current_features = frame_features[-1]
            residual = torch.norm(current_features - predicted_bg, p=2, dim=1, keepdim=True)
            residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
            results['reconstruction_residual'] = residual_norm

        return results

class BackgroundModelingFactory:

    @staticmethod
    def create_model(method_name, feature_channels=128, num_frames=5, **kwargs):
\
\
\
\
\
\
\
\
\
\
\

        method_name = method_name.lower()

        if method_name == 'temporal_median':
            return TemporalMedianBG(feature_channels, num_frames)

        elif method_name == 'gmm':
            num_gaussians = kwargs.get('num_gaussians', 3)
            return GaussianMixtureBG(feature_channels, num_frames, num_gaussians)

        elif method_name == 'lstm':
            hidden_dim = kwargs.get('hidden_dim', 256)
            return LSTMBackgroundBG(feature_channels, num_frames, hidden_dim)

        elif method_name == '3dcnn':
            return CNN3DBackgroundBG(feature_channels, num_frames)

        elif method_name == 'neural_field' or method_name == 'snf':
            from .neural_background_field import NeuralBackgroundField
            pos_freq = kwargs.get('pos_freq', 6)
            return NeuralBackgroundField(feature_channels, num_frames, pos_freq)

        else:
            raise ValueError(f"Unknown background modeling method: {method_name}")

    @staticmethod
    def get_available_methods():

        return ['temporal_median', 'gmm', 'lstm', '3dcnn', 'neural_field']
