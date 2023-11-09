"""
Preprocessing scripts to generate Pytorch DataLoaders for multi-object-datasets. 
Adapted from https://github.com/pemami4911/EfficientMORL/blob/main/lib/datasets.py 
"""

import math
import warnings
from typing import List
import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import random


DATA_ROOT_PATH = "data"


DATASET_IMG_RESOLUTION = {
    "clevr": (96, 96),  # Note(astanic): (96, 96) is the original clevr resolution in EMORL
    "multi_dsprites": (64, 64),  # Note(astanic): (64, 64) is the original dsprites resolution in EMORL
    "tetrominoes": (32, 32),
}


EMORL_DATASET_MAPPING = {
    'clevr': {
        'train': 'clevr6_with_masks_train.h5',
        'test': 'clevr6_with_masks_and_factors_test.h5'
    },
    'multi_dsprites': {
        'train': 'multi_dsprites_colored_on_grayscale.h5',
        'test': 'multi_dsprites_colored_on_grayscale_test.h5'
    },
    'tetrominoes': {
        'train': 'tetrominoes.h5',
        'test': 'tetrominoes_test.h5'
    },
}


class EMORLHdF5Dataset(torch.utils.data.Dataset):
    """
    The .h5 dataset is assumed to be organized as follows:
    {train|val|test}/
        imgs/  <-- a tensor of shape [dataset_size,H,W,C]
        masks/ <-- a tensor of shape [dataset_size,num_objects,H,W,C]
        factors/  <-- a tensor of shape [dataset_size,...]
    """
    def __init__(
        self,
        dataset_name: str, 
        split: str,
        use_32x32_res: bool,
        n_objects_cutoff: int,
        cutoff_type: str,
    ):    
        super(EMORLHdF5Dataset, self).__init__()
        h5_name = EMORL_DATASET_MAPPING[dataset_name][split]
        self.h5_path = str(Path(DATA_ROOT_PATH, h5_name))
        self.dataset_name = dataset_name
        if use_32x32_res:
            self.img_resolution = (32, 32)
        else:
            self.img_resolution = DATASET_IMG_RESOLUTION[self.dataset_name]
        self.n_channels = 3
        self.split = split
        self.n_objects_cutoff = n_objects_cutoff
        self.cutoff_type = cutoff_type

    def preprocess_image(
        self,
        img: np.ndarray,
        bg_mask: np.ndarray,
        ) -> np.ndarray:
        """
        img is assumed to be an array of integers each in 0-255 
        We preprocess them by mapping the range to -1,1
        
        """

        PIL_img = Image.fromarray(np.uint8(img))
        if self.dataset_name == "tetrominoes":
            PIL_img = PIL_img.resize(self.img_resolution)
        elif self.dataset_name == "multi_dsprites":
            PIL_img = PIL_img.resize(self.img_resolution)
        # square center crop of 192 x 192
        elif self.dataset_name == 'clevr':
            PIL_img = PIL_img.crop((64,29,256,221))
            PIL_img = PIL_img.resize(self.img_resolution)

        # H,W,C --> C,H,W
        img = np.transpose(np.array(PIL_img), (2,0,1))

        # image range is 0,1
        img = img / 255. # to [0,1]
        return img

    def preprocess_mask(
        self, 
        mask: np.ndarray
        ) -> np.ndarray:
        """
        [num_objects, h, w, c]
        Returns the square mask of size 192x192
        """
        o,h,w,c = mask.shape
        masks = []
        for i in range(o):
            mask_ = mask[i,:,:,0]
            PIL_mask = Image.fromarray(mask_, mode="F")
            if self.dataset_name == "tetrominoes":
                PIL_mask = PIL_mask.resize(self.img_resolution)
            elif self.dataset_name == "multi_dsprites":
                PIL_mask = PIL_mask.resize(self.img_resolution)
            elif self.dataset_name == "clevr":
                # square center crop of 192 x 192
                PIL_mask = PIL_mask.crop((64,29,256,221))
                # This resize was added such that we don't have to upsample predicted image in the evaluation
                # This also makes the learned phase evaluation easier.
                # Originaly Emami was resizing the predicted labels at evaluation time:
                # https://github.com/pemami4911/EfficientMORL/blob/main/eval.py#L336-L347
                PIL_mask = PIL_mask.resize(self.img_resolution, Image.NEAREST)
            masks += [np.array(PIL_mask)[...,None]]
        mask = np.stack(masks)  # [o,h,w,c]
        mask = np.transpose(mask, (0,3,1,2))  # [o,c,h,w]
        return mask

    def get_gt_phase_mask_from_labels(
        self,
        label: np.ndarray
    ) -> np.ndarray:
        """
        Converts the ground-truth label (class label) masks to ground-truth 
        phase space values.
        """
        class_labels = np.unique(label)
        # using the range of [-0.5*pi, 0.5*pi] for phases to avoid circularity issues
        gt_phase_values = np.linspace(-0.5*np.pi, 0.5*np.pi, len(class_labels))
        phase_mask = np.zeros_like(label, dtype=np.float32)
        for i in range(len(class_labels)):
            idxs = np.where(label == class_labels[i])
            phase_mask[idxs] = gt_phase_values[i]
        # adding a little noise in phase values 
        phase_noise_dist = np.random.uniform(-0.02*np.pi, 0.02*np.pi, size=phase_mask.shape)
        phase_mask += phase_noise_dist
        return phase_mask

    def get_gt_phase_mask_from_labels_fixed_order_by_spatial_locations(
        self,
        label: np.ndarray
    ) -> np.ndarray:
        """
        Converts the ground-truth label (class label) masks to ground-truth 
        phase space values in a fixed order:
            * determines the "average" spatial location of each object
            * assigns them the labels in ascending order, going left to right
        The idea behind this is to make the problem "identifiable" for the network.
        """
        # Use the max number of objects we have in any of our datasets
        # Doing this on purpose - because we want to have the possible values the phase can take to be the _same_ for every example
        n_class_labels = 6
        # Determine the order of masks:
        #   * background will always be the smallest phase
        #   * then we give the first phase value to

        # Create a linspace X grid
        img_w = label.shape[0]
        x_grid = np.linspace(0, img_w-1, img_w)
        x_grid = np.expand_dims(x_grid, axis=0)
        x_grid = np.repeat(x_grid, img_w, axis=0)

        # get one hot labels representation
        masks = np.eye(len(np.unique(label)))[label]
        centers = []
        for i in range(1, masks.shape[-1]):
            coords_1 = masks[:,:,i] * x_grid
            coords_1 = coords_1.astype(int)
            center = coords_1[np.nonzero(coords_1)].mean()
            centers.append(center)
        phase_order = np.argsort(centers)
        # Handle background
        phase_order += 1
        phase_order = [0] + list(phase_order)

        # Now simply do the same as the above function - the only change is _which_ phase value we assign
        # using the range of [-0.5*pi, 0.5*pi] for phases to avoid circularity issues
        gt_phase_values = np.linspace(-0.5*np.pi, 0.5*np.pi, n_class_labels)
        phase_mask = np.zeros_like(label, dtype=np.float32)
        for i in range(len(np.unique(label))):
            idxs = np.where(label == i)
            phase_mask[idxs] = gt_phase_values[phase_order[i]]
        # adding a little noise in phase values 
        phase_noise_dist = np.random.uniform(-0.02*np.pi, 0.02*np.pi, size=phase_mask.shape)
        phase_mask += phase_noise_dist
        return phase_mask

    def __len__(self) -> int:
        with h5py.File(self.h5_path,  'r') as data:
            data_size, _, _, _ = data[self.split]['imgs'].shape
            return data_size

    def __getitem__(self, i: int) -> dict:
        with h5py.File(self.h5_path,  'r') as data:
            imgs = data[self.split]['imgs'][i].astype('float32')

            masks = data[self.split]['masks'][i].astype('float32')
            outs = {}
            outs['images'] = self.preprocess_image(img=imgs, bg_mask=masks[0,:,:,0]).astype('float32')

            masks = self.preprocess_mask(masks)
            # n_objects, 1, img_h, img_w (6, 1, 64, 64)
            # gt-segmentation masks for coloured datasets must be of shape (h, w)
            masks = np.squeeze(masks, axis=1)
            # the first element of the first axis is background label
            masks_argmax = np.argmax(masks, axis=0)
            
            # Check if we should filter out this sample based on the number of objects
            # If yes, then sample a new index from the dataset at random
            # Note: we can only do samples filtering here (and not at the start of the function)
            # because of cropping (in the case of CLEVR), the number of objects might be different
            # in the ground truth and the cropped mask
            if self.n_objects_cutoff > 0:
                n_objects = len(np.unique(masks_argmax)) - 1  # subtract 1 for background
                if self.cutoff_type == 'eq':
                    if n_objects != self.n_objects_cutoff:
                        return self.__getitem__(np.random.randint(0, self.__len__()))
                elif self.cutoff_type == 'leq':
                    if n_objects > self.n_objects_cutoff:
                        return self.__getitem__(np.random.randint(0, self.__len__()))
                elif self.cutoff_type == 'geq':
                    if n_objects < self.n_objects_cutoff:
                        return self.__getitem__(np.random.randint(0, self.__len__()))
                else:
                    raise ValueError(f"Unknown n_objects cutoff type: {self.cutoff_type}")

            # find all overlaps and label them as -1. Note: the issue of correct object assignment
            # when 2 or more objects overlap should not exist in RGB data.
            masks_overlap = masks[1:] / 255
            masks_overlap = masks_overlap.sum(axis=0)
            outs['masks_argmax'] = masks_argmax
            outs['masks_overlap'] = masks_overlap
            masks_argmax[masks_overlap > 1] ==  -1
            outs['labels'] = masks_argmax
                
            return outs


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, num_workers, seed):
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(seed)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return data_loader


def random_split(dataset, lengths, generator=torch.default_generator):
    """
    local version of torch.utils.data.random_split() for backward compatibility to 1.10
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [torch.utils.data.dataset.Subset(dataset, indices[offset - length : offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
