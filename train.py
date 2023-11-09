from pathlib import Path
import random
import string
from tqdm import tqdm
import time
from itertools import cycle
from datetime import datetime

import torch
import numpy as np
import wandb

import data
import utils
import utils_args
import utils_contrastive

from models import ctcae_model

parser = utils_args.ArgumentParser()

# Training
parser.add_argument('-max_iterations', type=int, default=100_000,
                    help='Maximum number of training steps.')
parser.add_argument('-learning_rate', type=float, default=0.0004,
                    help='Learning rate.')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Mini-batch size.')
parser.add_argument('-num_workers', type=int, default=4,
                    help='Number of DataLoader workers.')
# Model
parser.add_argument('-phase_init', type=str, choices=["zero", "random", "positional"],
                    default="zero", help='Type of phase map initialization for CAE Input Layer')
parser.add_argument('-enc_n_out_channels', type=str, default='128,128,256,256,256',
                    help='Tuple in a string format denoting the number of output channels of the encoder layers. The decoder mirrors this.')
parser.add_argument('-enc_strides', type=str, default='2,1,2,1,2',
                    help='Tuple in a string format denoting the conv layer strides of the encoder. The decoder mirrors this.')
parser.add_argument('-enc_kernel_sizes', type=str, default='3,3,3,3,3',
                    help='Tuple in a string format denoting the conv layer strides of the encoder. The decoder mirrors this.')
parser.add_argument('-d_linear', type=int, default=256,
                    help='Number of hidden units in the encoder linear mapping.')
parser.add_argument('-decoder_type', type=str, choices=["conv_upsample", "conv_transpose"],
                    default="conv_upsample", help='Type of conv decoder architecture variation.')
parser.add_argument('-use_out_conv', type=int, default=0, 
                    help='Whether to use (1) 1x1 output convolution layer on Decoder or not (0).')
parser.add_argument('-use_out_sigmoid', type=int, default=0,
                    help='Whether to use (1) sigmoid activation on Decoder output or not (0).')

# Encoder contrastive loss parameters
parser.add_argument('-enc_n_anchors_to_sample', type=int, default=0,
                    help='Number of anchors to sample for the contrastive loss calculation.')
parser.add_argument('-enc_n_positive_pairs', type=int, default=1,
                    help='Number of positive pairs gathered for the contrastive loss calculation.')
parser.add_argument('-enc_contrastive_top_k', type=int, default=1,
                    help='Number of top (best) matching elements from which to select positives for the contrastive loss calculation.')
parser.add_argument('-enc_n_negative_pairs', type=int, default=2,
                    help='Number of negative pairs gathered for the contrastive loss calculation.')
parser.add_argument('-enc_contrastive_bottom_m', type=int, default=2,
                    help='Number of bottom (worst) matchine elements from which to select negatives for the contrastive loss calculation.')
# Decoder contrastive loss parameters
parser.add_argument('-dec_n_anchors_to_sample', type=int, default=100,
                    help='Number of anchors to sample for the contrastive loss calculation.')
parser.add_argument('-dec_n_positive_pairs', type=int, default=1,
                    help='Number of positive pairs gathered for the contrastive loss calculation.')
parser.add_argument('-dec_contrastive_top_k', type=int, default=5,
                    help='Number of top (best) matching elements from which to select positives for the contrastive loss calculation.')
parser.add_argument('-dec_n_negative_pairs', type=int, default=100,
                    help='Number of negative pairs gathered for the contrastive loss calculation.')
parser.add_argument('-dec_contrastive_bottom_m', type=int, default=500,
                    help='Number of bottom (worst) matchine elements from which to select negatives for the contrastive loss calculation.')
parser.add_argument('-dec_use_patches', type=bool, default=0,
                    help='Whether to divide (1) or not (0) the phase/magnitude maps into patches for contrasting.')
parser.add_argument('-dec_patch_size', type=int, default=2,
                    help='Size of patches (square-shaped) to be used for contrasting.')
# Loss coefficients
parser.add_argument('-loss_coeff_contrastive', type=float, default=0.,
                    help='Loss coefficient (weight in the final sum) of the contrastive loss.')
parser.add_argument('-contrastive_temperature', type=float, default=0.05,
                    help=' Softmax temperature (initialization) in the contrastive loss.')
# Data
parser.add_argument('-dataset_name', type=str, choices=["multi_dsprites", "tetrominoes", "clevr"], default="multi_dsprites",
                    help='Dataset name.')
parser.add_argument('-use_32x32_res', type=bool, default=1,
                    help='Whether to use 32x32 resolution (1) or default (tetr:32,32;dsp:64,64,clvr:96,96) (0).')
parser.add_argument('-n_objects_cutoff', type=int, default=0,
                    help='Number of objects by which to filter samples in the `getitem` function of the dataset loader.')
parser.add_argument('-cutoff_type', type=str, choices=["eq", "leq", "geq"],
                    default="eq", help='Cuttoff type: take only samples that equal, less-or-equal, greater-or-equal number of objects as specified by n_objects_cutoff.')

# Eval/Logging
parser.add_argument('-root_dir', type=str, default="results",
                    help='Root directory to save logs, ckpts, load data etc.')
parser.add_argument('-log_interval', type=int, default=10_000,
                    help='Logging interval (in steps).')
parser.add_argument('-eval_interval', type=int, default=10_000,
                    help='Evaluation interval (in steps).')
parser.add_argument('-eval_only_n_batches', type=int, default=0,
                    help='Evaluate only on a part of the validation set (for debugging).')
parser.add_argument('-use_wandb', type=int, default=1,
                    help='Flag to log results on wandb (1) or not (0).')
parser.add_argument('-use_cuda', type=int, default=1,
                    help='Use GPU acceleration (1) or not (0).')
parser.add_argument('-phase_mask_threshold', type=float, default=0.1,
                    help='Threshold on minimum magnitude to use when evaluating phases (CAE model); -1: no masking.')
parser.add_argument('-seed', type=int, default=0,
                    help='Random seed.')
parser.add_argument('-n_images_to_log', type=int, default=10,
                    help='Set to 0 to disable image rendering and logging. Number of images to take from a batch to generate the image grid logs.')
parser.add_argument('-plot_resize_resolution', type=int, default=256,
                    help='Resolution to which to resize the images before plotting. This is introduced due to the fine-level details in matplotlib plots.')


parser.add_profile([


    ################################################################
    # Base profile
    ################################################################

    parser.Profile('ctcae', {
        'decoder_type': 'conv_upsample',
        'use_out_conv': 0,
        'use_out_sigmoid': 0,
        'loss_coeff_contrastive': 0.0001,
        'enc_n_anchors_to_sample': 4,
    }),

    ################################################################
    # Tetrominoes 32x32 profiles
    ################################################################

    # CtCAE
    parser.Profile('ctcae_tetrominoes', {
        'max_iterations': 50_000,
        'dataset_name': 'tetrominoes',
    }, include='ctcae'),

    # CAE++
    parser.Profile('caepp_tetrominoes', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_tetrominoes'),

    # CAE
    parser.Profile('cae_tetrominoes', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_tetrominoes'),

    ################################################################
    # multi_dsprites 32x32 profiles
    ################################################################

    # CtCAE
    parser.Profile('ctcae_dsprites', {
        'dataset_name': 'multi_dsprites',
    }, include='ctcae'),

    # CAE++
    parser.Profile('caepp_dsprites', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_dsprites'),

    # CAE
    parser.Profile('cae_dsprites', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_dsprites'),

    ################################################################
    # CLEVR 32x32 profiles
    ################################################################

    # CtCAE
    parser.Profile('ctcae_clevr', {
        'dataset_name': 'clevr',
    }, include='ctcae'),

    # CAE++
    parser.Profile('caepp_clevr', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_clevr'),

    # CAE
    parser.Profile('cae_clevr', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_clevr'),

    ################################################################
    # multi_dsprites full resolution (64, 64) profiles
    ################################################################

    # CtCAE
    parser.Profile('ctcae_dsprites_64x64', {
        'use_32x32_res': 0,
        # Add one stride 1 layer and one stride 2 layer to the encoder and decoder
        # to keep the same hidden layer size
        'enc_n_out_channels': '128,128,256,256,256,256,256',
        'enc_strides': '2,1,2,1,2,1,2',
        'enc_kernel_sizes': '3,3,3,3,3,3,3',
        # Decoder CL params (since output res. is (64, 64) vs (32, 32))
        'dec_use_patches': 1,
        'dec_patch_size': 2,
    }, include='ctcae_dsprites'),

    # CAE++
    parser.Profile('caepp_dsprites_64x64', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_dsprites_64x64'),

    # CAE
    parser.Profile('cae_dsprites_64x64', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_dsprites_64x64'),

    ################################################################
    # CLEVR full resolution (96, 96) profiles
    ################################################################

    # CtCAE
    parser.Profile('ctcae_clevr_96x96', {
        'use_32x32_res': 0,
        # Add one stride 1 layer and one stride 2 layer to the encoder and decoder
        # to keep similar hidden layer size
        'enc_n_out_channels': '128,128,256,256,256,256,256',
        'enc_strides': '2,1,2,1,2,1,2',
        'enc_kernel_sizes': '3,3,3,3,3,3,3',
        # Adjust Encoder CL params (since we have 6x6 enc output)
        'enc_n_anchors_to_sample': 8,
        'enc_n_positive_pairs': 1,
        'enc_contrastive_top_k': 1,
        'enc_n_negative_pairs': 16,
        'enc_contrastive_bottom_m': 24,
        # Adjust Decoder CL params (since output res. is (96, 96) vs (32, 32))
        'dec_use_patches': 1,
        'dec_patch_size': 3,
    }, include='ctcae_clevr'),

    # CAE++
    parser.Profile('caepp_clevr_96x96', {
        'loss_coeff_contrastive': 0,
    }, include='ctcae_clevr_96x96'),

    # CAE
    parser.Profile('cae_clevr_96x96', {
        'loss_coeff_contrastive': 0,
        'use_out_conv': 1,
        'use_out_sigmoid': 1,
        'decoder_type': 'conv_transpose',
    }, include='ctcae_clevr_96x96'),

])

args = parser.parse_args()


torch.multiprocessing.set_sharing_strategy('file_system')

def build_datasets(args):
    """Function to build train, [validation] and test datasets loaders and initializers."""
    print(f"Loading and preprocessing data .....")

    print(f'Building {args.dataset_name} dataset.')
    if args.dataset_name in data.EMORL_DATASET_MAPPING:
        train_dataset = data.EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='train', 
            use_32x32_res=bool(args.use_32x32_res),
            n_objects_cutoff=args.n_objects_cutoff,
            cutoff_type=args.cutoff_type,
        )
        # EMORL ds don't have an exclusive val split -> split train into 2 exclusive parts
        train_dataset, _ = data.random_split(
            train_dataset, [0.95, 0.05], torch.Generator().manual_seed(args.seed)
        )

        val_dataset = data.EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='train', 
            use_32x32_res=bool(args.use_32x32_res),
            n_objects_cutoff=0,
            cutoff_type='',
        )
        _, val_dataset = data.random_split(
            val_dataset, [0.95, 0.05], torch.Generator().manual_seed(args.seed)
        )

        test_dataset = data.EMORLHdF5Dataset(
            dataset_name=args.dataset_name, 
            split='test', 
            use_32x32_res=bool(args.use_32x32_res),
            n_objects_cutoff=0,
            cutoff_type='',
        )
    else:
        raise ValueError(f'Unknown dataset {args.dataset_name}.')

    # Improve reproducibility in dataloader. (borrowed from Loewe)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = data.get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    val_dataloader = data.get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    test_dataloader = data.get_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    n_channels = test_dataset.n_channels

    return train_dataloader, val_dataloader, test_dataloader, n_channels



def build_model(args, n_channels):
    """Function to build model of requisite type and initialize optimizer."""

    if bool(args.use_32x32_res):
        img_resolution = (32, 32)
    else:
        img_resolution = data.DATASET_IMG_RESOLUTION[args.dataset_name]

    model = ctcae_model.ComplexAutoEncoderV2(
        img_resolution=img_resolution,
        n_in_channels=n_channels,
        # Architecture HPs
        enc_n_out_channels=args.enc_n_out_channels,
        enc_strides=args.enc_strides,
        enc_kernel_sizes=args.enc_kernel_sizes,
        d_linear=args.d_linear,
        decoder_type=args.decoder_type,
        use_out_conv=bool(args.use_out_conv),
        use_out_sigmoid=bool(args.use_out_sigmoid),
        phase_init=args.phase_init,
        # CL HPs
        use_contrastive_loss=bool(args.loss_coeff_contrastive > 0),
        contrastive_temperature=args.contrastive_temperature,
        # CL Encoder HPs
        enc_n_anchors_to_sample=args.enc_n_anchors_to_sample,
        enc_n_positive_pairs=args.enc_n_positive_pairs,
        enc_n_negative_pairs=args.enc_n_negative_pairs,
        enc_contrastive_top_k=args.enc_contrastive_top_k,
        enc_contrastive_bottom_m=args.enc_contrastive_bottom_m,
        # CL Decoder HPs
        dec_n_anchors_to_sample=args.dec_n_anchors_to_sample,
        dec_n_positive_pairs=args.dec_n_positive_pairs,
        dec_n_negative_pairs=args.dec_n_negative_pairs,
        dec_contrastive_top_k=args.dec_contrastive_top_k,
        dec_contrastive_bottom_m=args.dec_contrastive_bottom_m,
        dec_use_patches=args.dec_use_patches,
        dec_patch_size=args.dec_patch_size,
        seed=args.seed,
    )

    if args.use_cuda:
        model = model.cuda()
    
    utils.print_model_size(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-8
    )

    return model, optimizer

def compute_loss(model_inputs, model_outputs, args, contrastive_temperature):
    losses_dict = {}
    loss_total = 0
    loss_image_rec = torch.nn.functional.mse_loss(
        model_outputs["reconstruction"], model_inputs["images"], reduction="mean"
    )
    losses_dict['loss_rec'] = loss_image_rec
    loss_total += loss_image_rec

    # Contrastive loss
    if args.loss_coeff_contrastive > 0 and 'anchors' in model_outputs:
        assert 'positive_pairs' in model_outputs
        assert 'negative_pairs' in model_outputs
        assert len(model_outputs['anchors']) == len(model_outputs['positive_pairs']) == len(model_outputs['negative_pairs'])
        losses_dict['loss_contrastive'] = 0
        for i in range(len(model_outputs['anchors'])):
            # Iterate over all contrastive loss components (that stem from different layers of the model)
            anchors = model_outputs['anchors'][i]
            positive_pairs = model_outputs['positive_pairs'][i]
            negative_pairs = model_outputs['negative_pairs'][i]
            loss_contrastive = utils_contrastive.contrastive_soft_target_loss(
                anchors=anchors,
                positive_pairs=positive_pairs,
                negative_pairs=negative_pairs,
                temperature=contrastive_temperature,
            )
            losses_dict['loss_contrastive'] += loss_contrastive
            loss_total += loss_contrastive * args.loss_coeff_contrastive

    losses_dict['loss'] = loss_total
    return loss_total, losses_dict


def train_step(model_inputs, model, optimizer, args, step_number):
    optimizer.zero_grad()
    model_outputs = model(model_inputs["images"], step_number=step_number)
    loss, losses_dict = compute_loss(model_inputs, model_outputs, args, model.contrastive_temperature)
    loss.backward()
    optimizer.step()
    model_outputs.update(losses_dict)
    return model_outputs


def evaluation(split, step, args, eval_loader, model):
    model.eval()
    metrics_recorder = model.get_metrics_recorder()
    test_time = time.time()
    eval_iterator = iter(eval_loader)
    with torch.no_grad():
        for i_batch in tqdm(range(len(eval_loader))):
            eval_inputs = next(eval_iterator)
            if args.use_cuda:
                eval_inputs["images"] = eval_inputs["images"].cuda()

            outputs = model(eval_inputs["images"], step_number=i_batch)
            _, losses_dict = compute_loss(eval_inputs, outputs, args, model.contrastive_temperature)
            outputs.update(losses_dict)
            metrics_recorder.step(args, eval_inputs, outputs)
            if args.eval_only_n_batches > 0 and i_batch >= args.eval_only_n_batches:
                print(f'Stopping evaluation after {args.eval_only_n_batches} batches')
                break
    time_spent = time.time() - test_time
    extra_metrics = {
        'contrastive_temperature': model.get_contrastive_temperature_scalar(),
    }
    utils.log_results(
        split=split, use_wandb=args.use_wandb, step=step, time_spent=time_spent, metrics=metrics_recorder.log(), extra_metrics=extra_metrics)
    model.train()

    return metrics_recorder


def main(args):

    # set randomness
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb init
    if bool(args.use_wandb):
        run = wandb.init(config=args, project="cae-cscs-test", entity="idsia-olympics")
        run._settings.update(settings={'_service_wait':3000})
        
    # set logs and ckpts directories
    if bool(args.use_wandb):
        log_dir = Path(wandb.run.dir)
    else:
        run_name_id = str(''.join(random.choices(string.ascii_lowercase, k=5)))
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S_%f")
        run_name_id = run_name_id + timestamp
        log_dir = Path(args.root_dir) / args.dataset_name / args.model / run_name_id

    ckpt_dir = Path(log_dir / "ckpts")
    # create logs and ckpt directories
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    train_dataloader, val_dataloader, test_dataloader, n_channels = build_datasets(args)

    # BUILD MODEL
    model, optimizer = build_model(args, n_channels)

    # TRAINING LOOP
    try:
        print("-" * 89)
        print(f"Starting training for max {args.max_iterations} steps.")
        print(f"Number of batches in epoch={len(train_dataloader)}, batch size={args.batch_size} -> dataset size ~ {len(train_dataloader)*args.batch_size}.")
        print(f"At any point you can hit Ctrl + C to break out of the training loop early, but still evaluate on the test set.")

        print(f"Training model .....")
        step_start_time = time.time()
        train_iterator = cycle(train_dataloader)
        early_stop_score_best = model.get_metrics_recorder().get_init_value_for_early_stop_score()
        for step in range(1, args.max_iterations + 1):  # Adding one such that we evaluate the last step too
            step_start_time = time.time()

            # TRAIN STEP
            train_inputs = next(train_iterator)
            if args.use_cuda == 1:
                train_inputs["images"] = train_inputs["images"].cuda()
            train_outputs = train_step(train_inputs, model, optimizer, args, step_number=step)

            # LOGGING
            if step % args.log_interval == 0:
                metrics_recorder = model.get_metrics_recorder()
                metrics_recorder.step(args, train_inputs, train_outputs)
                step_time = time.time() - step_start_time
                extra_metrics = {
                    'lr': optimizer.param_groups[0]['lr'],
                    'contrastive_temperature': model.get_contrastive_temperature_scalar(),
                }
                utils.log_results(
                    split='train', use_wandb=args.use_wandb, step=step, time_spent=step_time,
                    metrics=metrics_recorder.log(), extra_metrics=extra_metrics,
                )

            # EVAL LOOP
            if step % args.eval_interval == 0:
                # Make sure to save the model in case evaluation NaNs out
                wandb_id = "-" + wandb.run.id if bool(args.use_wandb) else ""
                ckpt_fname = f"model-latest{wandb_id}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_outputs["loss"],
                }, ckpt_dir / ckpt_fname)

                eval_metrics_recorder = evaluation(
                    split='val', step=step, args=args, eval_loader=val_dataloader, 
                    model=model
                )                
                # CHECKPOINT MODEL
                ckpt_fname = ""
                early_stop_score_current = eval_metrics_recorder.get_current_early_stop_score()
                if eval_metrics_recorder.early_stop_score_improved(early_stop_score_current, early_stop_score_best):
                    print(f'Early stop / best model score improved from {early_stop_score_best} to {early_stop_score_current}.')
                    early_stop_score_best = early_stop_score_current
                    ckpt_fname = f"model-best{wandb_id}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_outputs["loss"],
                    }, ckpt_dir / ckpt_fname)
                
    except KeyboardInterrupt:
        print(f"-" * 89)
        print(f"KeyboardInterrupt signal received. Exiting early from training.")

    # TEST SET
    evaluation(
        split='test', step=args.max_iterations, args=args, eval_loader=test_dataloader,
        model=model
    )

    return 0


if __name__ == "__main__":
    main(args)
