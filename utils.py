from datetime import timedelta
import wandb

import numpy as np
import torch
import torch.nn.functional as F


def adjusted_rand_index(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor
    ) -> torch.Tensor:
    """
    Adapted from 
    https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), ("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32) 
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    _all_equal = lambda values: torch.all(torch.equal(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def log_results(split, use_wandb, step, time_spent, metrics, extra_metrics):
    # Console print
    console_log = f"{split} \t \t" \
        f"Step: {step} \t" \
        f"Time: {timedelta(seconds=time_spent)} \t" \
        f"MSE Loss: {metrics['loss']:.4e} \t"
    if 'ARI-FULL' in metrics:
        console_log += f"ARI-FULL: {metrics['ARI-FULL']:.4e} \t"
    if 'ARI-FG' in metrics:
        console_log += f"ARI-FG: {metrics['ARI-FG']:.4e} \t"
    print(console_log)

    # wandb log
    if bool(use_wandb):
        wandb_log = {
            'step': step,
            split + '/time_spent': time_spent,
        }
        wandb_log.update({f"{split}/{k}": v for k, v in extra_metrics.items()})
        wandb_log.update({f"{split}/{k}": v for k, v in metrics.items()})
        wandb.log(wandb_log)


def print_model_size(model):
    line_len = 89
    line_len2 = 25
    print('-' * line_len)
    # Native pytorch
    try:
        print(model)
    except:
        print('Warning: could not print the Native PyTorch model info - probably some module is `None`.')

    # One-by-one layer
    print('-' * line_len)
    print("Model params:")
    total_params = 0
    module_name = ""
    module_n_params = 0
    for name, param in model.named_parameters():
        if name.find('.') != -1:
            if module_name == "":
                module_name = name[:name.index('.')]
            if module_name != name[:name.index('.')]:
                print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')
                module_name = name[:name.index('.')]
                module_n_params = 0
        else:
            if module_name == "":
                module_name = name
            if module_name != name:
                print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')
                module_name = name
                module_n_params = 0
        n_params = np.prod(param.size())
        module_n_params += n_params
        print(f"\t {name} {n_params:,}")
        total_params += n_params
    print('=' * line_len2, module_name, f"{module_n_params:,}", '=' * line_len2, '\n')

    # Total Number of params
    print('-' * line_len)
    print(f"Total number of params: {total_params:,}")
    print('-' * line_len)

