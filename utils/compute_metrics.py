import math
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score

# Evaluate summary metrics (kTau and sRho)
def evaluate_summary(pred_score, gt_score, mask):
    for i in range(len(pred_score)):
        end_index = int(np.where(mask[i]==True)[0][-1]) + 1
        pred_score[i] = pred_score[i][:end_index]
        gt_score[i] = gt_score[i][:end_index]
    
    ktau_list, srho_list = [], []
    for i in range(len(pred_score)):
        ktau = kendalltau(pred_score[i], gt_score[i])[0]
        srho = spearmanr(pred_score[i], gt_score[i])[0]
        ktau_list.append(ktau)
        srho_list.append(srho)
    return np.mean(ktau_list), np.mean(srho_list)

# Helper function to calculate average precision for a single video
def _calculate_ap_for_video(pred_score, gt_score, rho, shot_duration_seconds=5, fps=1):
    shot_length_frames = int(shot_duration_seconds * fps)
    num_frames = len(pred_score)
    num_shots = math.ceil(num_frames / shot_length_frames)
    padding_size = num_shots * shot_length_frames - num_frames

    pred_padded = np.pad(pred_score, (0, padding_size), 'constant')
    gt_padded = np.pad(gt_score, (0, padding_size), 'constant')
    pred_shot_scores = np.mean(pred_padded.reshape(-1, shot_length_frames), axis=1)
    gt_shot_scores = np.mean(gt_padded.reshape(-1, shot_length_frames), axis=1)

    num_shots = len(gt_shot_scores)
    top_k = int(math.ceil(num_shots * rho))
    top_k_indices = np.argsort(gt_shot_scores)[-top_k:]
    gt_binary_labels = np.zeros(num_shots)
    gt_binary_labels[top_k_indices] = 1
    
    if np.sum(gt_binary_labels) == 0:
        return 0.0
    ap = average_precision_score(gt_binary_labels, pred_shot_scores)
    return ap

# Evaluate highlight metrics (mAP@50 and mAP@15)
def evaluate_highlight(pred_score_list, gt_score_list, mask, fps=1):
    for i in range(len(pred_score_list)):
        end_index = int(np.where(mask[i]==True)[0][-1]) + 1
        pred_score_list[i] = pred_score_list[i][:end_index]
        gt_score_list[i] = gt_score_list[i][:end_index]

    ap50_list, ap15_list = [], []
    for pred_score, gt_score in zip(pred_score_list, gt_score_list):
        ap50 = _calculate_ap_for_video(pred_score, gt_score, rho=0.50, fps=fps)
        ap15 = _calculate_ap_for_video(pred_score, gt_score, rho=0.15, fps=fps)
        ap50_list.append(ap50)
        ap15_list.append(ap15)

    map50 = np.mean(ap50_list) * 100
    map15 = np.mean(ap15_list) * 100
    return map50, map15