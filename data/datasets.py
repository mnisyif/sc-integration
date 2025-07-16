from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from torchvision import transforms
import numpy as np

NUM_DATASET_WORKERS = 8

class GenericImageDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        # collect all jpg/png paths
        self.img_paths = []
        for d in data_dirs:
            self.img_paths += glob(os.path.join(d, '*.jpg'))
            self.img_paths += glob(os.path.join(d, '*.png'))
        self.img_paths.sort()
        # we’ll ignore any passed‐in transform and apply size‐aware cropping below

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = Image.open(image_path).convert('RGB')
        # PIL.Image.size → (width, height)
        w, h = image.size
        # crop each dim down to a multiple of 128
        if h % 128 != 0 or w % 128 != 0:
            h = h - (h % 128)
            w = w - (w % 128)
        # center‐crop to (height, width), then to tensor
        crop_and_to_tensor = transforms.Compose([
            transforms.CenterCrop((h, w)),
            transforms.ToTensor()
        ])
        return crop_and_to_tensor(image)

    def __len__(self):
        return len(self.img_paths)


def get_loader(train_dirs, test_dirs, batch_size, num_workers=NUM_DATASET_WORKERS):
    # build datasets using the above GenericImageDataset
    train_ds = GenericImageDataset(train_dirs)
    test_ds  = GenericImageDataset(test_dirs)

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader