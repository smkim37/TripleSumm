import torch.nn as nn
from models.blocks import MultiScaleTemporalBlock, CrossModalFusionBlock
from models.layers import SinusoidalTemporalPE, ModalityEmbedding

# TripleSumm Model
class TripleSumm(nn.Module):
    def __init__(
        self,
        visual_dim, text_dim, audio_dim, input_dim, hidden_dim,
        num_model_layers, num_mst_layers, num_cmf_layers,
        num_heads, dropout, window_size, max_seq_len, get_attn_weights
        ):
        super().__init__()
        assert (num_model_layers * num_mst_layers) == len(window_size), "window_size must match num_model_layers * num_mst_layers"
        
        self.num_model_layers = num_model_layers
        self.num_mst_layers = num_mst_layers
        self.num_cmf_layers = num_cmf_layers
        self.get_attn_weights = get_attn_weights

        self.visual_proj = nn.Linear(visual_dim, input_dim)
        self.text_proj = nn.Linear(text_dim, input_dim)
        self.audio_proj = nn.Linear(audio_dim, input_dim)

        self.visual_ln = nn.LayerNorm(input_dim)
        self.text_ln = nn.LayerNorm(input_dim)
        self.audio_ln = nn.LayerNorm(input_dim)
        
        self.temporal_pe = SinusoidalTemporalPE(input_dim, max_seq_len, dropout)
        self.modality_embedding = ModalityEmbedding(input_dim)

        self.temporal_block = nn.ModuleList([
            MultiScaleTemporalBlock(input_dim, num_heads, dropout, window_size[i])
            for i in range(num_model_layers * num_mst_layers)
        ])

        self.modality_block = nn.ModuleList([
            CrossModalFusionBlock(input_dim, num_heads, dropout)
            for _ in range(num_model_layers * num_cmf_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, visual, text, audio, mask=None):
        visual = self.visual_proj(visual)
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        
        visual = self.visual_ln(visual)
        text = self.text_ln(text)
        audio = self.audio_ln(audio)
        
        fusion = (visual + text + audio) / 3
        
        fusion = self.temporal_pe(fusion)
        visual = self.temporal_pe(visual)
        text = self.temporal_pe(text)
        audio = self.temporal_pe(audio)
        
        fusion = fusion + self.modality_embedding(fusion, modality_index=0)
        visual = visual + self.modality_embedding(visual, modality_index=1)
        text = text + self.modality_embedding(text, modality_index=2)
        audio = audio + self.modality_embedding(audio, modality_index=3)

        attn_weights_list = []
        for i in range(self.num_model_layers):
            # Shared Multi-Scale Temporal block
            for j in range(self.num_mst_layers):
                fusion, visual, text, audio = self.temporal_block[i * self.num_mst_layers + j](fusion, visual, text, audio, mask)
            
            # Cross-Modal Fusion block
            for j in range(self.num_cmf_layers):
                fusion, attn_weights = self.modality_block[i * self.num_cmf_layers + j](fusion, visual, text, audio)
            
            if self.get_attn_weights:
                attn_weights_list.append(attn_weights.detach())
        
        if mask is not None:
            fusion = fusion * mask.unsqueeze(-1).float()

        out = self.head(fusion).squeeze(-1)
        return out, attn_weights_list