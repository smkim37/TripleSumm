from torch import optim
from models import TripleSumm

# Build model based on configuration
def build_model(cfg):
    if cfg.model == 'triplesumm':
        model = TripleSumm(
            visual_dim=cfg.visual_dim,
            text_dim=cfg.text_dim,
            audio_dim=cfg.audio_dim,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_model_layers=cfg.num_model_layers,
            num_mst_layers=cfg.num_mst_layers,
            num_cmf_layers=cfg.num_cmf_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            window_size=cfg.window_size,
            max_seq_len=cfg.max_seq_len,
            get_attn_weights=cfg.get_attn_weights
        )
    return model

# Build optimizer
def build_optimizer(cfg, model):
    if cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
    return optimizer

# Build learning rate scheduler
def build_scheduler(cfg, optimizer):
    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.num_epochs,
            eta_min=0
        )
    else:
        scheduler = None
    return scheduler