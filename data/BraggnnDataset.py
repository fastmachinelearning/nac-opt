import logging
import random

import os
import h5py
import numpy as np
import torch
from skimage import measure
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader, Dataset


def clean_patch(p, center):
    w, h = p.shape
    cc = measure.label(p > 0)
    if cc.max() == 1:
        return p

    # logging.warn(f"{cc.max()} peaks located in a patch")
    lmin = np.inf
    cc_lmin = None
    for _c in range(1, cc.max() + 1):
        lmax = peak_local_max(p * (cc == _c), min_distance=1)
        if lmax.shape[0] == 0:
            continue  # single pixel component
        lc = lmax.mean(axis=0)
        dist = ((lc - center) ** 2).sum()
        if dist < lmin:
            cc_lmin = _c
            lmin = dist
    return p * (cc == cc_lmin)


class BraggNNDataset(Dataset):
    def __init__(self,data_folder, psz=11, rnd_shift=0, use="train", train_frac=0.8, test_frac=0.1):
        self.psz = psz
        self.rnd_shift = rnd_shift

        # peak_datapath= data_folder + f"peaks-exp4train-psz{psz}.hdf5"
        peak_datapath = os.path.join(data_folder, f"peaks-exp4train-psz{psz}.hdf5")
        with h5py.File(peak_datapath, "r") as h5fd:
        # with h5py.File("./data/peaks-exp4train-psz%d.hdf5" % psz, "r") as h5fd:
            total_samples = h5fd["peak_fidx"].shape[0]
            train_end = int(train_frac * total_samples)
            test_start = int((1 - test_frac) * total_samples)

            if use == "train":
                sti, edi = 0, train_end
            elif use == "validation":
                sti, edi = train_end, test_start
            elif use == "test":
                sti, edi = test_start, None
            else:
                logging.error(
                    f"Unsupported use: {use}. This class should be used for building training, validation, or test set"
                )

            mask = h5fd["npeaks"][sti:edi] == 1  # use only single-peak patches
            mask = mask & ((h5fd["deviations"][sti:edi] >= 0) & (h5fd["deviations"][sti:edi] < 1))

            self.peak_fidx = h5fd["peak_fidx"][sti:edi][mask]
            self.peak_row = h5fd["peak_row"][sti:edi][mask]
            self.peak_col = h5fd["peak_col"][sti:edi][mask]

        self.fidx_base = self.peak_fidx.min()
        # only loaded frames that will be used
        frames_datapath = os.path.join(data_folder, "frames-exp4train.hdf5")
        # with h5py.File("./data/frames-exp4train.hdf5", "r") as h5fd:
        with h5py.File(frames_datapath, "r") as h5fd:
            self.frames = h5fd["frames"][self.peak_fidx.min() : self.peak_fidx.max() + 1]

        self.len = self.peak_fidx.shape[0]

    def __getitem__(self, idx):
        _frame = self.frames[self.peak_fidx[idx] - self.fidx_base]
        if self.rnd_shift > 0:
            row_shift = np.random.randint(-self.rnd_shift, self.rnd_shift + 1)
            col_shift = np.random.randint(-self.rnd_shift, self.rnd_shift + 1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd = int(self.peak_row[idx]) + row_shift
        pcol_rnd = int(self.peak_col[idx]) + col_shift

        row_base = max(0, prow_rnd - self.psz // 2)
        col_base = max(0, pcol_rnd - self.psz // 2)

        crop_img = _frame[
            row_base : (prow_rnd + self.psz // 2 + self.psz % 2), col_base : (pcol_rnd + self.psz // 2 + self.psz % 2)
        ]
        # if((crop_img > 0).sum() == 1): continue # ignore single non-zero peak
        if crop_img.size != self.psz**2:
            c_pad_l = (self.psz - crop_img.shape[1]) // 2
            c_pad_r = self.psz - c_pad_l - crop_img.shape[1]

            r_pad_t = (self.psz - crop_img.shape[0]) // 2
            r_pad_b = self.psz - r_pad_t - crop_img.shape[0]

            logging.warn(f"sample {idx} touched edge when crop the patch: {crop_img.shape}")
            crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode="constant")
        else:
            c_pad_l, r_pad_t = 0, 0

        _center = np.array([self.peak_row[idx] - row_base + r_pad_t, self.peak_col[idx] - col_base + c_pad_l])
        crop_img = clean_patch(crop_img, _center)
        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
            feature = crop_img

        px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
        py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

        return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len


def setup_data_loaders_braggnn(batch_size, IMG_SIZE, aug=0, num_workers=4, pin_memory=False, prefetch_factor=2, data_folder= "./data/"):
    ds_train = BraggNNDataset(data_folder=data_folder, psz=IMG_SIZE, rnd_shift=aug, use="train")
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        pin_memory=pin_memory,
    )
   

    ds_valid = BraggNNDataset(data_folder=data_folder, psz=IMG_SIZE, rnd_shift=0, use="validation")
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        pin_memory=pin_memory,
    )

    ds_test = BraggNNDataset(data_folder=data_folder, psz=IMG_SIZE, rnd_shift=0, use="test")
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return dl_train, dl_valid, dl_test
