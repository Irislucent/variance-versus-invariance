import os
import shutil
import argparse
import random
from glob import glob
from tqdm import tqdm
from copy import deepcopy


def split_train_val_test(data_dir, output_dir, val_percentage=0.1, test_percentage=0.1):
    """
    data_dir: directory directly containing wav files
    output_dir: directory to store the split data
    val_percentage: percentage of data to be used as validation set
    test_percentage: percentage of data to be used as test set
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    npy_paths = glob(os.path.join(data_dir, "*.npy"))
    n_val = int(len(npy_paths) * val_percentage)
    n_test = int(len(npy_paths) * test_percentage)
    print(f"{len(npy_paths)} wav files found.")
    random.shuffle(npy_paths)
    for i, npy_path in tqdm(enumerate(npy_paths)):
        if i < n_val:
            shutil.copy(npy_path, val_dir)
        elif i >= n_val and i < n_test + n_val:
            shutil.copy(npy_path, test_dir)
        elif i >= n_test + n_val:
            shutil.copy(npy_path, train_dir)

    print(
        f"Splited into {len(glob(os.path.join(train_dir, '*.npy')))} training files and {len(glob(os.path.join(val_dir, '*.npy')))} validation files and {len(glob(os.path.join(test_dir, '*.npy')))} test files."
    )


def split_ood_few_shot(data_dir, output_dir, n_shots=[1, 5, 10]):
    """
    data_dir: directory directly containing .npy files
    shots: number of shots for each class. The class label (style label) is the last word of the file name.
    files selected for few-shot learning will not be shown in the test set.
    """
    npy_paths = glob(os.path.join(data_dir, "*.npy"))
    test_npy_paths = deepcopy(npy_paths)
    for n_shot in n_shots:
        shots = {}
        shot_dir = os.path.join(output_dir, f"{n_shot}_shot")
        os.makedirs(shot_dir, exist_ok=True)
        for png_path in npy_paths:
            png_name = os.path.basename(png_path)
            s_label = png_name.split("_")[0]
            if s_label not in shots:
                shots[s_label] = []
            if len(shots[s_label]) < n_shot:
                shots[s_label].append(png_path)
                if png_path in test_npy_paths:
                    test_npy_paths.remove(png_path)
            else:
                continue
        for s_label, paths in shots.items():
            for path in paths:
                shutil.copy(path, shot_dir)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    for png_path in test_npy_paths:
        shutil.copy(png_path, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/insnotes_npy_")
    parser.add_argument("--save_dir", type=str, default="../data/InsNotes")
    parser.add_argument("--val_percentage", type=float, default=0.1)
    parser.add_argument("--test_percentage", type=float, default=0.1)
    parser.add_argument("--split_for_ood", action="store_true")

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    val_percentage = args.val_percentage
    test_percentage = args.test_percentage
    split_for_ood = args.split_for_ood

    if split_for_ood:
        split_ood_few_shot(data_dir, save_dir, n_shots=[1, 5, 10])
    else:
        split_train_val_test(data_dir, save_dir, val_percentage, test_percentage)
