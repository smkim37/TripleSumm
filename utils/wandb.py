import wandb

# Set up Weights & Biases (wandb) logging
def setup_wandb(cfg):
    if cfg.wandb:
        project_name = f"{cfg.dataset}"
        # Initialize wandb run with the specified project name and configuration
        wandb.init(
            entity='TripleSumm',
            project=project_name,
            name=f"{cfg.model}-{cfg.exp_name}",
            config=vars(cfg),
            reinit=True
        )
        return wandb.run.url
    return None

# Log training and validation metrics to wandb
def wandb_training(cfg, train_results, val_results, epoch):
    if cfg.wandb:
        wandb.log({
            f"train/kTau": train_results['ktau'],
            f"train/sRho": train_results['srho'],
            f"train/mAP50": train_results['map50'],
            f"train/mAP15": train_results['map15'],
            f"train/Loss": train_results['loss'],
            f"val/kTau": val_results['ktau'],
            f"val/sRho": val_results['srho'],
            f"val/mAP50": val_results['map50'],
            f"val/mAP15": val_results['map15'],
            f"val/Loss": val_results['loss']
        }, step=epoch)

# Log summary metrics to wandb
def wandb_summary(cfg, train_results, val_results, test_results, fold=None):
    if cfg.wandb:
        wandb.summary.update({
            "train/kTau": train_results['ktau'],
            "train/sRho": train_results['srho'],
            "train/mAP50": train_results['map50'],
            "train/mAP15": train_results['map15'],
            "val/kTau": val_results['ktau'],
            "val/sRho": val_results['srho'],
            "val/mAP50": val_results['map50'],
            "val/mAP15": val_results['map15']
        })
        wandb.log({
            "test/kTau": test_results['ktau'],
            "test/sRho": test_results['srho'],
            "test/mAP50": test_results['map50'],
            "test/mAP15": test_results['map15']
        })
        wandb.finish()