import torch
import os
from PIL import Image


def get_image_list(raw_image_path, clear_image_path, is_train=True):
    raw_filenames = sorted(os.listdir(raw_image_path))
    if is_train:
        clear_filenames = sorted(os.listdir(clear_image_path))
        return [
            (
                os.path.join(raw_image_path, rf),
                os.path.join(clear_image_path, cf),
                rf
            )
            for rf, cf in zip(raw_filenames, clear_filenames)
        ]
    else:
        return [
            (os.path.join(raw_image_path, rf), None, rf)
            for rf in raw_filenames
        ]


class UWNetDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image), self.transform(clear_image), "_"
        return self.transform(raw_image), "_", image_name

    def __len__(self):
        return len(self.image_list)
