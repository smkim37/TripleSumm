import os
import h5py
import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# MoSu and Mr. HiSum dataset class to load data and provide it to the model during training and testing
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        if cfg.dataset == 'mosu':
            split_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_split.json')
            visual_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_visual_clip.h5')
            text_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_text_roberta.h5')
            audio_path = os.path.join(cfg.data_dir, cfg.dataset, 'mosu_feat_audio_ast.h5')
        
        elif cfg.dataset == 'mrhisum':
            split_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_split.json')
            visual_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_visual_inceptionv3.h5')
            text_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_text_roberta.h5')
            audio_path = os.path.join(cfg.data_dir, cfg.dataset, 'mrhisum_feat_audio_ast.h5')
        
        self.video_ids = json.load(open(split_path, 'r'))[f'{split}_keys']
        self.gt_data = h5py.File(os.path.join(cfg.data_dir, cfg.dataset, f'{cfg.dataset}_gt.h5'), 'r')
        self.visual_data = h5py.File(visual_path, 'r')
        self.text_data = h5py.File(text_path, 'r')
        self.audio_data = h5py.File(audio_path, 'r')
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        data = {}
        video_id = self.video_ids[idx]
        data['video_id'] = video_id
        
        data['visual_feat'] = torch.Tensor(self.visual_data[video_id][...])
        data['text_feat'] = torch.Tensor(self.text_data[video_id][...])
        data['audio_feat'] = torch.Tensor(self.audio_data[video_id][...])
        
        data['gt_score'] = torch.Tensor(self.gt_data[video_id]['gt_score'][...])
        data['mask'] = torch.ones(data['gt_score'].shape[0], dtype=torch.bool)

        data['gt_summary'] = self.gt_data[video_id]['gt_summary'][...]
        data['change_points'] = self.gt_data[video_id]['change_points'][...]
        data['n_frames'] = int(data['visual_feat'].shape[0])
        data['picks'] = np.array([i for i in range(data['n_frames'])])
        return data

# Collate function to pad variable-length sequences in a batch
class CollateFn:
    def __call__(self, batch):
        data = {}
        data['video_id'] = [item['video_id'] for item in batch]
        
        data['visual_feat'] = pad_sequence([item['visual_feat'] for item in batch], batch_first=True, padding_value=0.0)
        data['text_feat'] = pad_sequence([item['text_feat'] for item in batch], batch_first=True, padding_value=0.0)
        data['audio_feat'] = pad_sequence([item['audio_feat'] for item in batch], batch_first=True, padding_value=0.0)
        
        data['gt_score'] = pad_sequence([item['gt_score'] for item in batch], batch_first=True, padding_value=0.0)
        data['mask'] = pad_sequence([item['mask'] for item in batch], batch_first=True, padding_value=0.0)
        
        data['gt_summary'] = [item['gt_summary'] for item in batch]
        data['change_points'] = [item['change_points'] for item in batch]
        data['n_frames'] = [item['n_frames'] for item in batch]
        data['picks'] = [item['picks'] for item in batch]
        return data