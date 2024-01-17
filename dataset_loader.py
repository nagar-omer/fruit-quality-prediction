import torch
from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import os
import numpy as np
from utils import read_sample
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import cv2


class FruitsDataset(Dataset):
    def __init__(self, path, transform=True):
        self._classes = ["Pasul", "A", "AA", "Muvhar"]
        self._id_to_classname = {c: i for i, c in enumerate(self._classes)}

        samples = glob(f"{path}/**/*.png")
        self._samples = pd.DataFrame([{"filename": f, "label_name": os.path.basename(os.path.dirname(f))}
                                      for f in samples])
        self._samples["label"] = self._samples.label_name.map(self._id_to_classname)
        self._samples = self._samples.sample(frac=1)

        self._transform = self._set_transform() if transform else None

    def _set_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[15.748789382616142, 13.131464734316049, 14.08085347716439],
                std=[13.631365106241876, 12.306106570865788, 16.544574399691726])
        ])

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        filename, label_name, label = self._samples.iloc[item][["filename", "label_name", "label"]]
        sample = self._samples.iloc[item]["sample"] if "sample" in self._samples.columns else read_sample(filename)

        # drop blanks
        sample = sample[np.where(sample.reshape(sample.shape[0], -1).var(axis=1) > 0)]
        if self._transform is not None:
            sample = torch.stack([self._transform(Image.fromarray(s)) for s in sample])
        else:
            sample = torch.Tensor(sample)

        on_hot_label = torch.zeros(4)
        on_hot_label[label] = 1
        return sample, on_hot_label


def fruit_collate_batch(batch):
    # input x - N x K x 3 x 28 x 28
    # input y - N x 4
    # k should be the same for all samples

    x, y = zip(*batch)
    # get k
    max_k = max([s.shape[0] for s in x])
    pad_sample = lambda s: np.pad(s, ((0, max_k - s.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
    x = np.stack([pad_sample(s) for s in x])
    return torch.Tensor(x), torch.vstack(y)


if __name__ == '__main__':
    ds = FruitsDataset(path="/Users/omernagar/Documents/Projects/fruit-quality-prediction/data")
    s = ds[1]
