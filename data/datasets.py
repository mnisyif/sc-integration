# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import os
# from glob import glob
# from torchvision import transforms
# import numpy as np

# NUM_DATASET_WORKERS = 8

# class GenericImageDataset(Dataset):
#     def __init__(self, data_dirs, transform=None):
#         # collect all jpg/png paths
#         self.img_paths = []
#         for d in data_dirs:
#             self.img_paths += glob(os.path.join(d, '*.jpg'))
#             self.img_paths += glob(os.path.join(d, '*.png'))
#         self.img_paths.sort()
#         # we'll ignore any passed‐in transform and apply size‐aware cropping below

#     def __getitem__(self, idx):
#         image_path = self.img_paths[idx]
#         image = Image.open(image_path).convert('RGB')
#         # PIL.Image.size → (width, height)
#         w, h = image.size
#         # crop each dim down to a multiple of 128
#         if h % 128 != 0 or w % 128 != 0:
#             h = h - (h % 128)
#             w = w - (w % 128)
#         # center‐crop to (height, width), then to tensor
#         crop_and_to_tensor = transforms.Compose([
#             transforms.CenterCrop((h, w)),
#             transforms.ToTensor()
#         ])
#         return crop_and_to_tensor(image)

#     def __len__(self):
#         return len(self.img_paths)


# def get_loader(train_dirs, test_dirs, batch_size, num_workers=NUM_DATASET_WORKERS):
#     # build datasets using the above GenericImageDataset
#     train_ds = GenericImageDataset(train_dirs)
#     test_ds  = GenericImageDataset(test_dirs)

#     def worker_init_fn(worker_id):
#         np.random.seed(42 + worker_id)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True,
#         worker_init_fn=worker_init_fn
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=False
#     )

#     return train_loader, test_loader

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
from torchvision import transforms
import numpy as np

NUM_DATASET_WORKERS = 8


class GenericImageDataset(Dataset):
    """
    Expects each directory in `data_dirs` to follow:
        root/<split>/<class_name>/*.jpg|*.png
    and returns (image_tensor, label) where
        label is an `int` in [0, num_classes-1].
    """
    def __init__(self, data_dirs, target_size=256):
        self.samples = []          # list of (img_path, class_idx)
        self.class_to_idx = {}     # str -> int
        self.idx_to_class = []     # int -> str (optional convenience)
        # self.num_classes = len(self.idx_to_class)

        # ‑‑ Build the mapping and sample list
        for root_dir in data_dirs:
            # find immediate sub‑dirs (= class names)
            for class_dir in sorted(os.listdir(root_dir)):
                abs_class_dir = os.path.join(root_dir, class_dir)
                if not os.path.isdir(abs_class_dir):
                    continue

                # assign a class index if new
                if class_dir not in self.class_to_idx:
                    self.class_to_idx[class_dir] = len(self.idx_to_class)
                    self.idx_to_class.append(class_dir)

                class_idx = self.class_to_idx[class_dir]

                # add all images inside this class folder
                for ext in ("*.jpg", "*.png"):
                    for img_path in glob(os.path.join(abs_class_dir, ext)):
                        self.samples.append((img_path, class_idx))

        self.samples.sort()  # deterministic order
        # ignore any passed‑in transform; we'll do size‑aware cropping below
        self.tx = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(target_size),           # or RandomResizedCrop
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # image = Image.open(img_path).convert("RGB")

        img = Image.open(img_path).convert("RGB")
        img = self.tx(img)
        return img, label
        # # PIL.Image.size → (width, height)
        # w, h = image.size
        # h -= (h % 128)
        # w -= (w % 128)

        # # Compose transform: center‑crop then ToTensor
        # xform = transforms.Compose([
        #     transforms.CenterCrop((h, w)),
        #     transforms.ToTensor()
        # ])

        # image_tensor = xform(image)
        # return image_tensor, label  # tuple

    @property
    def num_classes(self):
        return len(self.idx_to_class)

def get_loader(train_dirs, test_dirs, batch_size, num_workers=NUM_DATASET_WORKERS):

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)

    train_ds = GenericImageDataset(train_dirs)
    test_ds  = GenericImageDataset(test_dirs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn )
    
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, test_loader
