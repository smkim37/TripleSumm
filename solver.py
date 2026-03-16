import os
import h5py
import torch
import itertools
import numpy as np
from tqdm import tqdm

from utils.wandb import wandb_training
from utils.logger import setup_logger, log_training
from utils.compute_metrics import evaluate_summary, evaluate_highlight
from models import build_model, build_optimizer, build_scheduler

class Solver:
    def __init__(self, cfg, train_loader, val_loader, test_loader):
        self.cfg = cfg
        self.logger = setup_logger('solver', self.cfg.output_dir, overwrite=True)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.model = build_model(self.cfg).to(self.cfg.device)
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        
        self.criterion = torch.nn.MSELoss()
    
    # Training method for training the model and evaluating on train/val sets
    def train(self):
        best_model_score = -np.inf
        patience_counter = 0
        
        train_results = self.evaluate(split='train')
        val_results = self.evaluate(split='val')
        log_training(self.logger, train_results, val_results, epoch=0)
        wandb_training(self.cfg, train_results, val_results, epoch=0)
        
        for epoch in tqdm(range(1, self.cfg.num_epochs + 1), desc='Training', leave=False):
            self.model.train()
            
            # Iterate over training batches
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.cfg.num_epochs}', leave=False):
                visual = batch['visual_feat'].to(self.cfg.device, non_blocking=True)
                text = batch['text_feat'].to(self.cfg.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.cfg.device, non_blocking=True)
                
                gt_score = batch['gt_score'].to(self.cfg.device, non_blocking=True)
                mask = batch['mask'].to(self.cfg.device, non_blocking=True)

                output, _ = self.model(visual, text, audio, mask=mask)
                loss = self.criterion(output[mask], gt_score[mask])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Evaluation on train and val sets
            train_results = self.evaluate(split='train')
            val_results = self.evaluate(split='val')
            log_training(self.logger, train_results, val_results, epoch)
            wandb_training(self.cfg, train_results, val_results, epoch)

            # Best model check pointing & early stopping
            model_score = val_results['ktau'] + val_results['srho']
            if model_score > best_model_score:
                best_model_score = model_score
                patience_counter = 0
                best_model_ckpt = os.path.join(self.cfg.output_dir, 'best_model_ckpt.pth')
                torch.save(self.model.state_dict(), best_model_ckpt)
                self.logger.info(f"\t# New best model at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    self.logger.info(f"\t# Early stopping at epoch {epoch}")
                    break
    
    # Evaluation method for evaluating the model on train/val/test sets
    def evaluate(self, split='val'):
        self.model.eval()
        
        if split == 'train':
            loader = list(itertools.islice(self.train_loader, len(self.val_loader)))
        elif split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        
        ktau_list, srho_list = [], []
        map50_list, map15_list = [], []
        loss_list = []
        
        h5_file = None
        if self.cfg.get_attn_weights:
            attn_weights_path = os.path.join(os.path.dirname(self.cfg.model_ckpt), f'{split}_attn_weights.h5')
            h5_file = h5py.File(attn_weights_path, 'w')

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'Evaluating {split} set', leave=False):
                video_ids = batch['video_id']
                
                visual = batch['visual_feat'].to(self.cfg.device, non_blocking=True)
                text = batch['text_feat'].to(self.cfg.device, non_blocking=True)
                audio = batch['audio_feat'].to(self.cfg.device, non_blocking=True)
                
                gt_score = batch['gt_score'].to(self.cfg.device, non_blocking=True)
                mask = batch['mask'].to(self.cfg.device, non_blocking=True)

                output, attn_weights = self.model(visual, text, audio, mask=mask)
                loss = self.criterion(output[mask], gt_score[mask])
                
                if output.dim() == 3:
                    output = output.squeeze(-1)
                
                pred_score = output.detach().cpu().numpy().tolist()
                gt_score = gt_score.detach().cpu().numpy().tolist()
                mask = mask.detach().cpu().numpy()
                
                ktau, srho = evaluate_summary(pred_score, gt_score, mask)
                map50, map15 = evaluate_highlight(pred_score, gt_score, mask)
                
                ktau_list.append(ktau)
                srho_list.append(srho)
                map50_list.append(map50)
                map15_list.append(map15)
                loss_list.append(loss.item())
                
                if self.cfg.get_attn_weights:
                    batch_size = visual.size(0)
                    for i in range(batch_size):
                        video_id = video_ids[i]
                        if video_id not in h5_file:
                            video_group = h5_file.create_group(video_id)
                            for layer_idx, layer_weights_tensor in enumerate(attn_weights):
                                sample_layer_weights = layer_weights_tensor[i].detach().cpu().numpy()
                                video_group.create_dataset(f'layer_{layer_idx}', data=sample_layer_weights)
        
        if self.cfg.get_attn_weights:
            h5_file.close()
        
        return {
            'ktau': np.mean(ktau_list),
            'srho': np.mean(srho_list),
            'map50': np.mean(map50_list),
            'map15': np.mean(map15_list),
            'loss': np.mean(loss_list)
        }