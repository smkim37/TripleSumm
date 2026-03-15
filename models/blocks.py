import torch
import torch.nn as nn
from models.layers import WindowedSelfAttention, CrossModalAttention, SwiGLU

# Windowed Self-Attention Layer
class WindowedSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, window_size):
        super().__init__()
        self.wsa = WindowedSelfAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size,
        )
        self.ffn = SwiGLU(input_dim=input_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.wsa(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

# Multi-Scale Temporal Block
class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, window_size):
        super().__init__()
        self.wsa_layer = WindowedSelfAttentionLayer(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )

    def forward(self, fusion, visual, text, audio, mask=None):
        fusion = self.wsa_layer(fusion, mask)
        visual = self.wsa_layer(visual, mask)
        text = self.wsa_layer(text, mask)
        audio = self.wsa_layer(audio, mask)
        return fusion, visual, text, audio

# Cross-Modal Attention Layer
class CrossModalAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.cma = CrossModalAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn = SwiGLU(input_dim=input_dim, dropout=dropout)
        self.norm_q = nn.LayerNorm(input_dim)
        self.norm_c = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        attn_out, attn_weights = self.cma(self.norm_q(query), self.norm_c(context))
        query = query + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(query))
        query = query + self.dropout(ffn_out)
        return query, attn_weights

# Cross-Modal Fusion Block
class CrossModalFusionBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.cma_layer = CrossModalAttentionLayer(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, fusion, visual, text, audio):
        query = fusion
        context = torch.stack((visual, text, audio), dim=2)

        B, T, D = query.shape
        _, _, N, _ = context.shape
        query_reshaped = query.reshape(B * T, 1, D)
        context_reshaped = context.reshape(B * T, N, D)

        query_update, attn_weights = self.cma_layer(query_reshaped, context_reshaped)
        fusion_update = query_update.reshape(B, T, D)
        attn_weights = attn_weights.squeeze(2).view(B, T, self.num_heads, N)
        return fusion_update, attn_weights