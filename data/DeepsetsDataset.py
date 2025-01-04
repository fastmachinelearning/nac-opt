import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Download the normalized_data3 folder with the data from: https://drive.google.com/drive/folders/1OrVEYa1JV8k6wtnq4qem1ZsMWLrAsYtx?usp=drive_link


class DeepSetDataset(Dataset):
    def __init__(self, data_files, target_files, use="train", train_frac=0.8, test_frac=0.1):
        self.data = []
        self.targets = []

        for data_file, target_file in zip(data_files, target_files):
            data = np.load(data_file)
            targets = np.load(target_file)
            self.data.append(data)
            self.targets.append(targets)

        total_samples = len(data_files)
        train_end = int(train_frac * total_samples)
        test_start = int((1 - test_frac) * total_samples)

        if use == "train":
            sti, edi = 0, train_end
        elif use == "validation":
            sti, edi = train_end, test_start
        elif use == "test":
            sti, edi = test_start, None
        else:
            raise ValueError(
                f"Unsupported use: {use}. This class should be used for building training, validation, or test set"
            )

        for data_file, target_file in zip(data_files[sti:edi], target_files[sti:edi]):
            data = np.load(data_file)
            targets = np.load(target_file)
            self.data.append(data)
            self.targets.append(targets)
        print(f"Loaded {len(self.data)} files for {use} set")

    def __getitem__(self, index):
        fold_idx = index // len(self.data[0])
        item_idx = index % len(self.data[0])
        data = self.data[fold_idx][item_idx]
        target = self.targets[fold_idx][item_idx]
        return torch.from_numpy(data).permute((1, 0)), torch.from_numpy(target)

    def __len__(self):
        return len(self.data) * len(self.data[0])


def setup_data_loaders_deepsets(base_file_name, batch_size=32, num_workers=4, pin_memory=False, prefetch_factor=2):
    train_data_files = [f"./data/normalized_data3/x_train_{base_file_name}.npy"]
    train_target_files = [f"./data/normalized_data3/y_train_{base_file_name}.npy"]
    test_data_files = [f"./data/normalized_data3/x_test_{base_file_name}.npy"]
    test_target_files = [f"./data/normalized_data3/y_test_{base_file_name}.npy"]

    ds_train = DeepSetDataset(train_data_files, train_target_files, use="train")
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        pin_memory=pin_memory,
    )

    ds_valid = DeepSetDataset(train_data_files, train_target_files, use="validation")
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        pin_memory=pin_memory,
    )

    ds_test = DeepSetDataset(test_data_files, test_target_files, use="test")
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
