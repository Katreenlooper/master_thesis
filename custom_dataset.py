from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, json_data, flip_transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.flip_transform = flip_transform
        self.total_imgs = []

        for frame in json_data:
            if frame['labels'] is not None:
                self.total_imgs.append(frame['name'])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = []
        tensor_image_flipped = []
        if self.transform:
            tensor_image = self.transform(image)
        if self.flip_transform:
            tensor_image_flipped = self.flip_transform(image)

        return tensor_image, tensor_image_flipped