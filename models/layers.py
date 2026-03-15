import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal Temporal Positional Encoding
class SinusoidalTemporalPE(nn.Module):
    def __init__(self, input_dim, max_seq_len, dropout):
        super().__init__()
        assert input_dim % 2 == 0, "input_dim must be even"
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim))
        pe = torch.zeros(1, max_seq_len, input_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

# Modality Embedding
class ModalityEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()  # 0: fusion, 1: visual, 2: text, 3: audio
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=input_dim)

    def forward(self, x, modality_index):
        batch_size, seq_len, _ = x.shape
        modality_tensor = torch.full((batch_size, seq_len), modality_index, dtype=torch.long, device=x.device)
        return self.embedding(modality_tensor)

# SwiGLU Feedforward Network
class SwiGLU(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        hidden_dim = int(input_dim * 2 * (2/3))
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        x = self.dropout(x)
        return x

# Windowed Self-Attention
class WindowedSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, window_size):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        assert window_size >= 0, "window_size must be non-negative"
        assert window_size % 2 == 1 or window_size == 0, "window_size must be odd or 0"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.use_window = window_size > 0
        self.window_size = window_size
        self.radius = window_size // 2

        self.qkv_proj = nn.Linear(input_dim, input_dim*3, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _edge_pad(x, radius):
        if radius == 0:
            return x
        left = x[:, :1, :].expand(-1, radius, -1)
        right = x[:, -1:, :].expand(-1, radius, -1)
        return torch.cat((left, x, right), dim=1)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        if self.use_window:  # Windowed self-attention
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = q.unsqueeze(-2)

            padded_k = self._edge_pad(k, self.radius)
            padded_k = padded_k.reshape(B, T + 2 * self.radius, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_windows = padded_k.unfold(2, self.window_size, 1)

            padded_v = self._edge_pad(v, self.radius)
            padded_v = padded_v.reshape(B, T + 2 * self.radius, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_windows = padded_v.unfold(2, self.window_size, 1)

            attn_scores = (q @ k_windows) * self.scale
            attn_scores = attn_scores.squeeze(-2)

            if mask is not None:
                mask_windows = (~mask).view(B, T, 1)
                padded_mask = F.pad(mask_windows, (0, 0, self.radius, self.radius), value=True)
                attn_mask = padded_mask.unfold(1, self.window_size, 1).squeeze(2).unsqueeze(1)
                attn_scores = attn_scores.masked_fill(attn_mask, -1e9)

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            v_windows_permuted = v_windows.permute(0, 1, 2, 4, 3)
            out = (attn_probs.unsqueeze(-2) @ v_windows_permuted).squeeze(-2)
            out = out.transpose(1, 2).reshape(B, T, D)
            out = self.out_proj(out)

        else:  # Global self-attention
            qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            scores = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn_mask = (~mask).view(B, 1, 1, T).expand(-1, self.num_heads, T, -1)
                scores = scores.masked_fill(attn_mask, -1e9)

            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            out = attn_probs @ v
            out = out.transpose(1, 2).reshape(B, T, D)
            out = self.out_proj(out)
        return out

# Cross-Modal Attention
class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        B_q, T_q, _ = query.shape
        B_c, T_c, _ = context.shape

        q = self.q_proj(query).reshape(B_q, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B_c, T_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B_c, T_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        scores = (q @ k.transpose(-2, -1)) * self.scale

        attn_probs = F.softmax(scores, dim=-1)
        attn_weights = attn_probs
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v
        out = out.transpose(1, 2).reshape(B_q, T_q, self.input_dim)
        out = self.out_proj(out)
        return out, attn_weights