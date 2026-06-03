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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .darknet import BaseConv
from .neural_background_field import NeuralBackgroundField
from .background_modeling import BackgroundModelingFactory
from .module.CBAM import CBAM
from .stst import SpatioTemporalSwinTransformer

class EarlyFusionModule(nn.Module):
\
\
\

    def __init__(self, channels=[128, 256, 512], output_dim=128):
        super().__init__()
        self.channels = channels
        self.output_dim = output_dim

        self.align_convs = nn.ModuleList()
        for i, c in enumerate(channels):
            if i == 0:
                self.align_convs.append(
                    BaseConv(c, output_dim, 3, 1)
                )
            else:
                upsample_factor = 2 ** i
                self.align_convs.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False),
                        BaseConv(c, output_dim, 3, 1)
                    )
                )

        self.fusion_conv = nn.Sequential(
            BaseConv(output_dim * len(channels), output_dim * 2, 3, 1),
            BaseConv(output_dim * 2, output_dim, 3, 1)
        )

        self.attention = CBAM(output_dim)

    def forward(self, multi_scale_feats):
\
\
\
\
\

        aligned_feats = []
        target_size = multi_scale_feats[0].shape[-2:]

        for i, feat in enumerate(multi_scale_feats):
            aligned_feat = self.align_convs[i](feat)

            if aligned_feat.shape[-2:] != target_size:
                aligned_feat = F.interpolate(
                    aligned_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            aligned_feats.append(aligned_feat)

        fused_feat = torch.cat(aligned_feats, dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        fused_feat = self.attention(fused_feat)

        return fused_feat

class AppearanceAnomalyExtractor(nn.Module):
\
\
\

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        self.feat_norm = nn.LayerNorm(feature_dim)

        self.anomaly_enhance = nn.Sequential(
            BaseConv(1, 32, 3, 1),
            BaseConv(32, 16, 3, 1),
            BaseConv(16, 1, 3, 1, act='sigmoid')
        )

    def cosine_distance_residual(self, real_feat, synth_feat):
\
\
\
\
\
\
\

        B, C, H, W = real_feat.shape

        real_flat = real_feat.view(B, C, -1).permute(0, 2, 1)
        synth_flat = synth_feat.view(B, C, -1).permute(0, 2, 1)

        real_norm = self.feat_norm(real_flat)
        synth_norm = self.feat_norm(synth_flat)

        real_norm = F.normalize(real_norm, p=2, dim=-1)
        synth_norm = F.normalize(synth_norm, p=2, dim=-1)

        cosine_sim = torch.sum(real_norm * synth_norm, dim=-1, keepdim=True)

        cosine_dist = 1.0 - cosine_sim

        residual_map = cosine_dist.permute(0, 2, 1).view(B, 1, H, W)

        return residual_map

    def forward(self, real_feat, synth_feat):
\
\
\
\
\
\
\

        raw_anomaly = self.cosine_distance_residual(real_feat, synth_feat)

        enhanced_anomaly = self.anomaly_enhance(raw_anomaly)

        return enhanced_anomaly

class CausalAnomalyNeck(nn.Module):
    def __init__(self,
                 channels=[128, 256, 512],
                 num_frame=5,
                 fusion_dim=128,
                 bg_method='neural_field',
                 bg_kwargs=None,
                 ):
        super().__init__()

        self.num_frame = num_frame
        self.fusion_dim = fusion_dim

        self.anomaly_threshold = 0.3
        self.anomaly_scale = 0.8

        self.early_fusion = EarlyFusionModule(channels, fusion_dim)

        self.small_target_enhancer = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, 1, 1, groups=fusion_dim//8),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(),
            nn.Conv2d(fusion_dim, fusion_dim, 1),
            nn.Sigmoid()
        )

        self.background_field = NeuralBackgroundField(
            feature_channels=fusion_dim,
            num_frames=num_frame,
            pos_freq=6
        )

        self.anomaly_extractor = AppearanceAnomalyExtractor(fusion_dim)

        self.stst_processor = SpatioTemporalSwinTransformer(
            num_frames=num_frame,
            embed_dim=32,
            num_heads=2,
            window_size=4,
            num_layers=1,
            drop_path_rate=0.2
        )

        self.feature_modulation = nn.Sequential(
            nn.Dropout2d(0.1),
            BaseConv(fusion_dim, fusion_dim, 3, 1),
            nn.Dropout2d(0.1),
            CBAM(fusion_dim)
        )

    def process_temporal_sequence(self, feat_sequence):
\
\
\
\
\
\
\

        bg_results = self.background_field(feat_sequence, return_residual=True)

        predicted_bg = bg_results['predicted_bg']
        current_feat = feat_sequence[-1]

        anomaly_maps = []
        for t in range(self.num_frame):
            curr_feat = feat_sequence[t]

            if t < self.num_frame - 1:

                temp_sequence = feat_sequence[:t+1] + [feat_sequence[t]] * (self.num_frame - t - 1)

                temp_bg_results = self.background_field(temp_sequence, return_residual=False)

                frame_bg = temp_bg_results['predicted_bg']
            else:
                frame_bg = predicted_bg

            anomaly_map = self.anomaly_extractor(curr_feat, frame_bg)
            anomaly_maps.append(anomaly_map)

        return current_feat, anomaly_maps

    def forward(self, feats):
\
\
\
\
\
\
\

        multi_scale_sequences = []
        for scale_idx in range(len(feats[0])):
            scale_sequence = [feats[frame_idx][scale_idx] for frame_idx in range(len(feats))]
            multi_scale_sequences.append(scale_sequence)

        fused_sequence = []
        for frame_idx in range(self.num_frame):
            frame_multi_scale = [seq[frame_idx] for seq in multi_scale_sequences]
            fused_feat = self.early_fusion(frame_multi_scale)
            fused_sequence.append(fused_feat)

        current_feat, anomaly_maps = self.process_temporal_sequence(fused_sequence)
        raw_anomaly = self.stst_processor(anomaly_maps)

        final_anomaly = torch.clamp(raw_anomaly, 0, 1)
        final_anomaly = torch.where(
            final_anomaly > self.anomaly_threshold,
            final_anomaly * self.anomaly_scale,
            final_anomaly * 0.1
        )
        small_target_mask = self.small_target_enhancer(current_feat)
        enhanced_feat = current_feat * small_target_mask

        final_feat = enhanced_feat * (1.0 + final_anomaly)

        final_feat = self.feature_modulation(final_feat)

        return [final_feat]
