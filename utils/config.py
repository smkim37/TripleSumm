import os
import yaml
import argparse

# Function to parse configuration
def get_config():
    parser = argparse.ArgumentParser(description="Configuration Parser")
    
    # General settings
    parser.add_argument('--exp_name', type=str, default='exp', help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode of operation')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='mosu', choices=['mosu', 'mrhisum'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory where datasets are stored')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Model settings
    parser.add_argument('--model', type=str, default='triplesumm', help='Model architecture to use')
    parser.add_argument('--visual_dim', type=int, default=768, help='Dimension of visual features')
    parser.add_argument('--text_dim', type=int, default=768, help='Dimension of text features')
    parser.add_argument('--audio_dim', type=int, default=768, help='Dimension of audio features')
    parser.add_argument('--input_dim', type=int, default=512, help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--num_model_layers', type=int, default=2, help='Number of model layers')
    parser.add_argument('--num_mst_layers', type=int, default=2, help='Number of multi-scale temporal layers per model layer')
    parser.add_argument('--num_cmf_layers', type=int, default=2, help='Number of cross-modal fusion layers per model layer')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--window_size', type=int, nargs='+', default=[5,15,45,0], help='Window sizes for multi-scale temporal blocks')
    parser.add_argument('--max_seq_len', type=int, default=10000, help='Maximum sequence length')
    parser.add_argument('--get_attn_weights', action='store_true', help='Whether to return attention weights from the model')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler to use')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for cosine scheduler with warmup')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    # Miscellaneous settings
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to a pre-trained model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb', action='store_true', help='Whether to use Weights & Biases for experiment tracking')
    
    # Load YAML config
    temp_args, _ = parser.parse_known_args()
    yaml_path = f"./configs/{temp_args.dataset}.yaml"
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = os.path.join('./outputs', args.dataset, args.model, args.exp_name)
    
    return args