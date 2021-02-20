import os

import numpy as np
import pandas as pd
import torch.utils.data as data
import yaml
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def get_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MaskBaseDataset(data.Dataset):
    class Labels:
        mask = 0
        incorrect = 1
        normal = 2

    _file_names = {
        "mask1.jpg": Labels.mask,
        "mask2.jpg": Labels.mask,
        "mask3.jpg": Labels.mask,
        "mask4.jpg": Labels.mask,
        "mask5.jpg": Labels.mask,
        "incorrect_mask.jpg": Labels.incorrect,
        "normal.jpg": Labels.normal
    }

    image_paths = []
    labels = []

    def __init__(self, img_root, meta_path, phase, mean=None, std=None):
        self.img_root = img_root
        self.meta_path = meta_path
        self.phase = phase
        self.mean = [0.5366, 0.5323, 0.5183]  # todo
        self.std = [0.2456, 0.2517, 0.2605]  # todo

        self.setup()
        self.calc_statistics()

    def setup(self):
        # -- Homework 1
        df = pd.read_csv(self.meta_path)  # "../upstage/metadata.csv"
        profiles = df.path.tolist()
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_root, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                if os.path.exists(img_path) and is_image_file(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def train_transform(self):
        return transforms.Compose([
            Resize((128, 96), Image.BILINEAR),  # todo : img_size
            RandomRotation([-8, +8]),
            GaussianBlur(51, (0.1, 2.0)),
            ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),  # todo : param
            ToTensor(),
            Normalize(mean=self.mean, std=self.std),
            AddGaussianNoise(0., 1.)
        ])

    def test_transform(self):
        return transforms.Compose([
            Resize((128, 96), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std),
        ])

    def transform(self, image):
        if self.phase == "train":
            transform = self.train_transform()
        elif self.phase == "test":
            transform = self.test_transform()
        else:
            raise AttributeError
        return transform(image)

    def __getitem__(self, index):
        image = self.read_image(index)
        label = self.get_label(index)

        image_transform = self.transform(image)
        return image_transform, label

    def __len__(self):
        return len(self.image_paths)

    def get_label(self, index):
        return self.labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def set_phase(self, phase):
        assert phase in ["train", "test"], "phase should be either train or test"
        self.phase = phase


class MaskMultiLabelDataset(MaskBaseDataset):
    class GenderLabels:
        male = 0
        female = 1

    gender_labels = []

    def setup(self):
        df = pd.read_csv(self.meta_path)  # "../upstage/metadata.csv"
        profiles = df.path.tolist()
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_root, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                if os.path.exists(img_path) and is_image_file(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    self.gender_labels.append(gender_label)

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def __getitem__(self, index):
        image = self.read_image(index)
        base_label = self.get_label(index)
        gender_label = self.get_gender_label(index)

        image_transform = self.transform(image)
        labels = np.array([base_label, gender_label])
        return image_transform, labels