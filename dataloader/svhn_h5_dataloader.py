"""
This is the dataloader module for SVHN dataset (HDF5).
The HDF5 file should be generated by preprocessing.
"""

import os
import h5py
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

S_LIST = []  # it's really hard to determine the styles of house front doors...

C_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class SVHNH5Dataset(Dataset):
    def __init__(
        self,
        data_path,
        n_fragments,
        fragment_len,
        portion=1,
        c_list=C_LIST,
        s_list=S_LIST,
    ):
        """
        data_path: path to the hdf5 file, with or without the extension
        n_fragments: number of fragments to cut from each sample (for this dataset, it should be 2)
        fragment_len: both the width and height of the resized image
        portion: portion of the dataset to use

        Implemented for hdf5 files.
        """
        self.data_path = data_path
        self.n_fragments = n_fragments
        self.fragment_len = fragment_len
        self.portion = portion
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.c_list = c_list
        self.s_list = s_list

        if not self.data_path.endswith(".hdf5"):
            self.data_path += ".hdf5"
        assert os.path.exists(self.data_path)

        with h5py.File(self.data_path, "r") as f:
            all_data = f["data"]
            all_content_label = f["content_label"]  # digits
            all_style_label = f[
                "style_label"
            ]  # the names of original images of cropped images

            # "classify" all cropped images to their origins
            self.image_lib = {}  # every key: value pair is like origin (style) name : list of [cropped images, content label]
            for i, origin_name in enumerate(all_style_label):
                if origin_name not in self.image_lib:
                    self.image_lib[origin_name] = []
                self.image_lib[origin_name].append([all_data[i], all_content_label[i]])
            self.image_origins = list(self.image_lib.keys())

        if portion != 1:
            self.image_origins = self.image_origins[
                : int(len(self.image_origins) * portion)
            ]
            self.image_origins = sorted(self.image_origins)

    def __len__(self):
        return len(self.image_origins)

    def __getitem__(self, idx):
        """
        returns a tuple of:
        - selected_images: a tensor of shape (n_fragments, 3, fragment_len, fragment_len)
        - selected_content_labels: a tensor of shape (n_fragments)
        - image_origin: a string representing the style
        """
        image_origin = self.image_origins[idx]
        images = [x[0] for x in self.image_lib[image_origin]]
        content_labels = [x[1] for x in self.image_lib[image_origin]]

        # randomly pick n_fragments fragments from the images
        n_images = len(images)
        if n_images > self.n_fragments:
            n_fragments = self.n_fragments
            selected_indices = random.sample(range(n_images), n_fragments)
            selected_images = [images[i] for i in selected_indices]
            selected_content_labels = [content_labels[i] for i in selected_indices]
        else:
            selected_images = images
            selected_content_labels = content_labels

        # transform the images
        selected_images = [self.transform(x) for x in selected_images]
        selected_images = torch.stack(selected_images)
        selected_content_labels = torch.tensor(selected_content_labels)

        return (
            selected_images,
            selected_content_labels,
            image_origin,  # note that this is a string
        )


def get_dataloader(
    data_path,
    batch_size,
    n_fragments=3,
    fragment_len=32,
    num_workers=0,
    portion=1,
    shuffle=True,
):
    dataset = SVHNH5Dataset(
        data_path=data_path,
        n_fragments=n_fragments,
        fragment_len=fragment_len,
        portion=portion,
        c_list=C_LIST,
        s_list=S_LIST,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
