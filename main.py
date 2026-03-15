import os
import torch
from solver import Solver
from dataset import Dataset, CollateFn

from utils.config import get_config
from utils.seed import set_seed, seed_worker
from utils.wandb import setup_wandb, wandb_summary
from utils.logger import setup_logger, log_config, log_dataset, log_model, log_results

# Main class for setting up the training and evaluation pipeline
class Main():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.wandb_url = setup_wandb(self.cfg)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.logger = setup_logger('main', self.cfg.output_dir, overwrite=True)
        self.cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_config(self.logger, self.cfg)
        
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        self.loader_args = {
            'batch_size': self.cfg.batch_size,
            'num_workers': self.cfg.num_workers,
            'collate_fn': CollateFn(),
            'worker_init_fn': seed_worker,
            'generator': g,
            'pin_memory': True
        }
    
    # Train method for training the model and evaluating on train, val, and test sets
    def train(self):
        train_dataset = Dataset(self.cfg, split='train')
        val_dataset = Dataset(self.cfg, split='val')
        test_dataset = Dataset(self.cfg, split='test')
        log_dataset(self.logger, train_dataset, val_dataset, test_dataset)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **self.loader_args)
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **self.loader_args)
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **self.loader_args)
        
        solver = Solver(self.cfg, train_loader, val_loader, test_loader)
        log_model(self.logger, self.cfg, solver.model)
        
        if self.cfg.model_ckpt is not None:
            solver.model.load_state_dict(torch.load(self.cfg.model_ckpt, map_location=self.cfg.device, weights_only=True))
            self.logger.info(f"Loaded model checkpoint from {self.cfg.model_ckpt}\n")
        
        solver.train()
        
        best_model_ckpt = os.path.join(self.cfg.output_dir, 'best_model_ckpt.pth')
        solver.model.load_state_dict(torch.load(best_model_ckpt, map_location=self.cfg.device, weights_only=True))
        
        train_results = solver.evaluate(split='train')
        val_results = solver.evaluate(split='val')
        test_results = solver.evaluate(split='test')

        log_results(self.logger, [train_results, val_results, test_results], ['train', 'val', 'test'])
        wandb_summary(self.cfg, train_results, val_results, test_results)
    
    # Test method for evaluating a trained model on the test set
    def test(self):
        test_dataset = Dataset(self.cfg, split='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **self.loader_args)
        log_dataset(self.logger, None, None, test_dataset)
        
        solver = Solver(self.cfg, None, None, test_loader)
        log_model(self.logger, self.cfg, solver.model)
        
        solver.model.load_state_dict(torch.load(self.cfg.model_ckpt, map_location=self.cfg.device, weights_only=True))
        self.logger.info(f"Loaded model checkpoint from {self.cfg.model_ckpt}\n")
        
        test_results = solver.evaluate(split='test')
        log_results(self.logger, [test_results], ['test'])

# Main entry point for the script
if __name__ == "__main__":
    cfg = get_config()
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')
    main = Main(cfg)
    if cfg.mode == 'train':
        main.train()
    elif cfg.mode == 'test':
        main.test()