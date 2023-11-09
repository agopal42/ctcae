from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from einops import rearrange, reduce
import wandb

import utils_plot


def clip_and_rescale(input_tensor, clip_value):
    if torch.is_tensor(input_tensor):
        clipped = torch.clamp(input_tensor, min=0, max=clip_value)
    elif isinstance(input_tensor, np.ndarray):
        clipped = np.clip(input_tensor, a_min=0, a_max=clip_value)
    else:
        raise NotImplementedError

    return clipped * (1 / clip_value)


def spherical_to_cartesian_coordinates(x):
    # Second dimension of x contains spherical coordinates: (r, phi_1, ... phi_n).
    num_dims = x.shape[1]
    out = torch.zeros_like(x)

    r = x[:, 0]
    phi = x[:, 1:]

    sin_component = 1
    for i in range(num_dims - 1):
        out[:, i] = r * torch.cos(phi[:, i]) * sin_component
        sin_component = sin_component * torch.sin(phi[:, i])

    out[:, -1] = r * sin_component
    return out


def phase_to_cartesian_coordinates(phase_mask_threshold, phase, norm_magnitude):
    # Map phases on unit-circle and transform to cartesian coordinates.
    unit_circle_phase = torch.cat(
        (torch.ones_like(phase)[:, None], phase[:, None]), dim=1
    )

    if phase_mask_threshold != -1:
        # When magnitude is < phase_mask_threshold, use as multiplier to mask out respective phases from eval.
        unit_circle_phase = unit_circle_phase * norm_magnitude[:, None]

    # cartesian_form torch.Size([64, 2, 3, 32, 32])
    cartesian_form = spherical_to_cartesian_coordinates(unit_circle_phase)
    return cartesian_form


def apply_kmeans(phase_mask_threshold, output_phase, output_norm_magnitude, labels_true, img_resolution, seed):

    input_phase = phase_to_cartesian_coordinates(phase_mask_threshold, output_phase, output_norm_magnitude)
    input_phase = input_phase.detach().cpu().numpy()
    input_phase = rearrange(input_phase, "b p c h w -> b h w (c p)")

    # num_clusters = int(torch.max(labels_true).item()) + 1
    num_clusters = int(len(torch.unique(labels_true)))

    batch_size, num_angles = output_phase.shape[0:2]
    labels_pred = (
        np.zeros((batch_size, img_resolution[0], img_resolution[1]))
        + num_clusters
    )
    # Run k-means on each image separately.
    cluster_metrics_batch = {
        'inter_cluster_min': 0.,
        'inter_cluster_max': 0.,
        'inter_cluster_mean': 0.,
        'inter_cluster_std': 0.,
        'intra_cluster_dist': 0.,
        'intra_cluster_dist_safe': 0.,
        'intra_cluster_n_nan': 0.,
    }
    n_objects = []
    for img_idx in range(batch_size):
        in_phase = input_phase[img_idx]
        # num_clusters_img = int(torch.max(labels_true[img_idx]).item()) + 1
        num_clusters_img = int(len(torch.unique(labels_true[img_idx])))
        n_objects.append(num_clusters_img - 1)  # -1 for background class

        # Remove areas in which objects overlap before k-means analysis.
        # filtering only for grayscale img
        if num_angles == 1: 
            label_idx = np.where(labels_true[img_idx].cpu().numpy() != -1)
            in_phase = in_phase[label_idx]
        # flatten image before running k-means
        else:
            in_phase = rearrange(in_phase, "h w c -> (h w) c")
        # Run k-means.
        k_means = KMeans(n_clusters=num_clusters_img, n_init=10, random_state=seed).fit(
            in_phase
        )

        # Calculate cluster inter- and intra-cluster distances.
        cluster_metrics = calculate_cluster_metrics(in_phase, k_means)
        for key in cluster_metrics_batch.keys():
            cluster_metrics_batch[key] += cluster_metrics[key]

        # Create result image: fill in k_means labels & assign overlapping areas to class zero.
        cluster_img = (
            np.zeros((img_resolution[0], img_resolution[1])) + num_clusters
        )
        # for grayscale img -> assign cluster idxs to only non-overlapping regions
        if num_angles == 1:
            cluster_img[label_idx] = k_means.labels_
        # for colour img -> assign cluster idxs to all regions
        else:
            cluster_img = np.reshape(
                k_means.labels_, (img_resolution[0], img_resolution[1])
            )
        labels_pred[img_idx] = cluster_img
    n_objects = np.array(n_objects)

    # Calculate average cluster metrics.
    for key in cluster_metrics_batch.keys():
        cluster_metrics_batch[key] /= batch_size
    return labels_pred, cluster_metrics_batch, n_objects


def calculate_cluster_metrics(in_phase, k_means):

        # Calculate inter-cluster distances.
        centroids = k_means.cluster_centers_
        c1 = np.expand_dims(centroids, axis=0)
        c2 = np.expand_dims(centroids, axis=1)
        diff = (c1 - c2) ** 2  # n_clusters x n_clusters x RGB * 2 (euclidean coordinates)
        diff = np.mean(diff, axis=-1)
        # Create mask to ignore diagonal elements.
        mask = np.ones(diff.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        # Calculate inter-cluster distance metrics.
        inter_cluster_min = np.min(diff[mask])
        inter_cluster_max = np.max(diff[mask])
        inter_cluster_mean = np.mean(diff[mask])
        inter_cluster_std = np.std(diff[mask])

        # Calculate intra-cluster distance.
        centroids_expanded = np.expand_dims(centroids, axis=0)
        centroids_expanded = np.repeat(centroids_expanded, in_phase.shape[0], axis=0)
        intra_cluster_dist = []
        intra_cluster_dist_safe = []
        intra_cluster_n_nan = []
        for i in range(centroids.shape[0]):  # number of clusters
            # Mask out all points that are not in cluster i.
            arg1 = in_phase[k_means.labels_ == i]
            arg2 = centroids_expanded[k_means.labels_ == i]
            diff = (arg1 - arg2[:, i]) ** 2
            intra_cluster_dist.append(np.mean(diff))
            intra_cluster_dist_safe.append(np.nan_to_num(np.mean(diff)))
            intra_cluster_n_nan.append(np.count_nonzero(np.isnan(diff)))
        intra_cluster_dist = np.mean(intra_cluster_dist)
        intra_cluster_dist_safe = np.mean(intra_cluster_dist_safe)
        intra_cluster_n_nan = np.mean(intra_cluster_n_nan)
        
        return {
            'inter_cluster_min': inter_cluster_min,
            'inter_cluster_max': inter_cluster_max,
            'inter_cluster_mean': inter_cluster_mean,
            'inter_cluster_std': inter_cluster_std,
            'intra_cluster_dist': intra_cluster_dist,
            'intra_cluster_dist_safe': intra_cluster_dist_safe,
            'intra_cluster_n_nan': intra_cluster_n_nan,
        }

def calc_ari_score(batch_size, labels_true, labels_pred, with_background):
    ari = 0
    per_sample_ari = []
    for idx in range(batch_size):
        if with_background:
            area_to_eval = np.where(
                labels_true[idx] > -1
            )  # Remove areas in which objects overlap.
        else:
            area_to_eval = np.where(
                labels_true[idx] > 0
            )  # Remove background & areas in which objects overlap.

        sample_ari = adjusted_rand_score(
            labels_true[idx][area_to_eval], labels_pred[idx][area_to_eval]
        )
        ari += sample_ari
        per_sample_ari.append(sample_ari)
    per_sample_ari = np.array(per_sample_ari)

    return ari / batch_size, per_sample_ari


def get_ari_for_n_objects(ari_w_bg_scores, ari_wo_bg_scores, n_objects, n_objects_max):
    ari_w_bg_per_n_objects = []
    ari_wo_bg_per_n_objects = []
    n_samples_per_n_objects = []
    for i in range(n_objects_max):
        idxs = n_objects == i
        n_samples_per_n_objects.append(np.sum(idxs))
        # take the sum because we want to average over all samples once we log the results
        ari_w_bg_per_n_objects.append(np.sum(ari_w_bg_scores[idxs]))
        ari_wo_bg_per_n_objects.append(np.sum(ari_wo_bg_scores[idxs]))
    return ari_w_bg_per_n_objects, ari_wo_bg_per_n_objects, n_samples_per_n_objects


@dataclass
class CAEMetricsRecorder:
    num_batches: int = 0
    loss_total: float = 0.
    loss_rec: float = 0.
    loss_contrastive: float = 0.
    ari_w_bg: float = 0.
    ari_wo_bg: float = 0.
    N_MAX_OBJECTS: int = 8  # In CLEVR max objects in the image is 7, dsprites it is 6.
    ari_w_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(CAEMetricsRecorder.N_MAX_OBJECTS)])
    ari_wo_bg_per_n_objects: List[float] = field(default_factory=lambda: [0. for _ in range(CAEMetricsRecorder.N_MAX_OBJECTS)])
    n_samples_per_n_objects: List[int] = field(default_factory=lambda: [0. for _ in range(CAEMetricsRecorder.N_MAX_OBJECTS)])
    inter_cluster_min: float = 0.
    inter_cluster_max: float = 0.
    inter_cluster_mean: float = 0.
    inter_cluster_std: float = 0.
    intra_cluster_dist: float = 0.
    intra_cluster_dist_safe: float = 0.
    intra_cluster_n_nan: float = 0.
    img_grid_all: Optional[wandb.Image] = None
    img_grid_main: Optional[wandb.Image] = None
    img_grid_enc_layers: Optional[wandb.Image] = None
    img_grid_dec_layers: Optional[wandb.Image] = None

    @staticmethod
    def get_init_value_for_early_stop_score():
        return float('-inf')
    
    def get_current_early_stop_score(self):
        return self.ari_wo_bg / self.num_batches
    
    def early_stop_score_improved(self, ari_current, ari_best):
        if ari_current > ari_best:
            return True
        return False
    
    def step(self, args, model_input, model_output):

        self.num_batches += 1
        self.loss_total += model_output["loss"].item()
        self.loss_rec += model_output["loss_rec"].item()
        if 'loss_contrastive' in model_output:
            self.loss_contrastive += model_output["loss_contrastive"].item()

        # Determine output phase predictions
        gt_labels = model_input["labels"]
        output_phase = model_output["complex_phase"]
        img_resolution = tuple(output_phase.shape[-2:])

        output_magnitude_scaled = clip_and_rescale(model_output["complex_magnitude"], args.phase_mask_threshold)

        output_labels_pred, cluster_metrics, n_objects = apply_kmeans(
            args.phase_mask_threshold, output_phase, output_magnitude_scaled, gt_labels,
            img_resolution, seed=args.seed
        )

        # ARI score
        ari_w_bg, ari_w_bg_per_sample = calc_ari_score(gt_labels.shape[0], gt_labels, output_labels_pred, with_background=True)
        ari_wo_bg, ari_wo_bg_per_sample = calc_ari_score(gt_labels.shape[0], gt_labels, output_labels_pred, with_background=False)
        self.ari_w_bg += ari_w_bg
        self.ari_wo_bg += ari_wo_bg
        # Calculate ARI score wrt to the number of objects in the image
        ari_w_bg_per_n_objects, ari_wo_bg_per_n_objects, n_samples_per_n_objects = get_ari_for_n_objects(
            ari_w_bg_per_sample, ari_wo_bg_per_sample, n_objects, n_objects_max=self.N_MAX_OBJECTS)
        for i in range(self.N_MAX_OBJECTS):
            if n_samples_per_n_objects[i] > 0:
                self.ari_w_bg_per_n_objects[i] += ari_w_bg_per_n_objects[i]
                self.ari_wo_bg_per_n_objects[i] += ari_wo_bg_per_n_objects[i]
                self.n_samples_per_n_objects[i] += n_samples_per_n_objects[i]

        # Cluster metrics
        self.inter_cluster_min += cluster_metrics["inter_cluster_min"]
        self.inter_cluster_max += cluster_metrics["inter_cluster_max"]
        self.inter_cluster_mean += cluster_metrics["inter_cluster_mean"]
        self.inter_cluster_std += cluster_metrics["inter_cluster_std"]
        self.intra_cluster_dist += cluster_metrics["intra_cluster_dist"]
        self.intra_cluster_dist_safe += cluster_metrics["intra_cluster_dist_safe"]
        self.intra_cluster_n_nan += cluster_metrics["intra_cluster_n_nan"]

        # Visual logs - rendering images is costly, so render them only once
        if self.img_grid_all is None and args.n_images_to_log > 0:
            input_images = model_input["images"][:args.n_images_to_log]
            reconstructed_images = model_output["reconstruction"][:args.n_images_to_log]
            gt_label = model_input["labels"][:args.n_images_to_log]
            pred_label = output_labels_pred[:args.n_images_to_log]

            # Final layer phase and magnitude
            phase = output_phase[:args.n_images_to_log]
            # for color img (3 output phases) -> viz average phase value
            if phase.shape[1] != 1:
                phase = reduce(phase, 'b c h w -> b 1 h w', 'mean')
            magnitude = model_output["complex_magnitude"][:args.n_images_to_log]
            magnitude = magnitude.detach().cpu().numpy()
            phase = phase.detach().cpu().numpy()
            plot_resize_resolution = (args.plot_resize_resolution, args.plot_resize_resolution)
            plot_radial, plot_groups_phase, plot_groups_magnitude = (
                utils_plot.create_phase_colorcoded_groupings_and_radial_plots(
                    plot_resize_resolution, phase, magnitude))

            # enc_layer_phase_maps
            # (B, N_layers, 1, 32, 32)
            enc_layer_phase_maps = model_output['enc_layer_phase_maps']
            enc_layer_magnitude_maps = model_output['enc_layer_magnitude_maps']
            n_enc_layers = enc_layer_phase_maps.shape[1]
            plot_enc_layer_radial = []
            plot_enc_layer_groups_phase = []
            plot_enc_layer_groups_magnitude = []
            for i in range(n_enc_layers):
                phase = enc_layer_phase_maps[:args.n_images_to_log, i]
                phase = phase.detach().cpu().numpy()
                magnitude = enc_layer_magnitude_maps[:args.n_images_to_log, i]
                magnitude = magnitude.detach().cpu().numpy()
                radial_plot, groups_plot_phase, groups_plot_magnitude = (
                    utils_plot.create_phase_colorcoded_groupings_and_radial_plots(
                        plot_resize_resolution, phase, magnitude))
                plot_enc_layer_radial.append(radial_plot)
                plot_enc_layer_groups_phase.append(groups_plot_phase)
                plot_enc_layer_groups_magnitude.append(groups_plot_magnitude)

            # dec_layer_phase_maps
            # (B, N_layers, 1, 32, 32)
            dec_layer_phase_maps = model_output['dec_layer_phase_maps']
            dec_layer_magnitude_maps = model_output['dec_layer_magnitude_maps']
            n_dec_layers = dec_layer_phase_maps.shape[1]
            plot_dec_layer_radial = []
            plot_dec_layer_groups_phase = []
            plot_dec_layer_groups_magnitude = []
            for i in range(n_dec_layers):
                phase = dec_layer_phase_maps[:args.n_images_to_log, i]
                phase = phase.detach().cpu().numpy()
                magnitude = dec_layer_magnitude_maps[:args.n_images_to_log, i]
                magnitude = magnitude.detach().cpu().numpy()
                radial_plot, groups_plot_phase, groups_plot_magnitude = (
                    utils_plot.create_phase_colorcoded_groupings_and_radial_plots(
                        plot_resize_resolution, phase, magnitude))
                plot_dec_layer_radial.append(radial_plot)
                plot_dec_layer_groups_phase.append(groups_plot_phase)
                plot_dec_layer_groups_magnitude.append(groups_plot_magnitude)

            outputs_plots = utils_plot.create_image_grids_for_logging(
                plot_resize_resolution=plot_resize_resolution,
                img_in=input_images,
                img_rec=reconstructed_images,
                gt_label=gt_label,
                pred_label=pred_label,
                plot_radial=plot_radial,
                plot_groups_phase=plot_groups_phase,
                plot_groups_magnitude=plot_groups_magnitude,
                plot_enc_layer_radial=plot_enc_layer_radial,
                plot_enc_layer_groups_phase=plot_enc_layer_groups_phase,
                plot_enc_layer_groups_magnitude=plot_enc_layer_groups_magnitude,
                plot_dec_layer_radial=plot_dec_layer_radial,
                plot_dec_layer_groups_phase=plot_dec_layer_groups_phase,
                plot_dec_layer_groups_magnitude=plot_dec_layer_groups_magnitude,
            )

            self.img_grid_all = utils_plot.tensor_to_wandb_image(outputs_plots['img_grid_all'])
            self.img_grid_main = utils_plot.tensor_to_wandb_image(outputs_plots['img_grid_main'])
            self.img_grid_enc_layers = utils_plot.tensor_to_wandb_image(outputs_plots['img_grid_enc_layers'])
            self.img_grid_dec_layers = utils_plot.tensor_to_wandb_image(outputs_plots['img_grid_dec_layers'])
    
    def log(self):
        logs = {
            "loss": self.loss_total / self.num_batches,
            "loss_rec": self.loss_rec / self.num_batches,
            "loss_contrastive": self.loss_contrastive / self.num_batches,
            "ARI-FULL": self.ari_w_bg / self.num_batches,
            "ARI-FG": self.ari_wo_bg / self.num_batches,
            "inter_cluster_min": self.inter_cluster_min / self.num_batches,
            "inter_cluster_max": self.inter_cluster_max / self.num_batches,
            "inter_cluster_mean": self.inter_cluster_mean / self.num_batches,
            "inter_cluster_std": self.inter_cluster_std / self.num_batches,
            "intra_cluster_dist": self.intra_cluster_dist / self.num_batches,
            "intra_cluster_dist_safe": self.intra_cluster_dist_safe / self.num_batches,
            "intra_cluster_n_nan": self.intra_cluster_n_nan / self.num_batches,
            # Visual logs
            "img_grid_all": self.img_grid_all,
            "img_grid_main": self.img_grid_main,
            "img_grid_enc_layers": self.img_grid_enc_layers,
            "img_grid_dec_layers": self.img_grid_dec_layers,
        }
        for i in range(1, len(self.n_samples_per_n_objects)):
            logs[f"ARI-FG-{i}"] = self.ari_wo_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"ARI-FULL-{i}"] = self.ari_w_bg_per_n_objects[i] / self.n_samples_per_n_objects[i] if self.n_samples_per_n_objects[i] > 0 else 0
            logs[f"n_images-w-{i}-objects"] = self.n_samples_per_n_objects[i]
        return logs
