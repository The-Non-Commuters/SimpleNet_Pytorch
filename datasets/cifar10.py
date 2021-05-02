from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import random


class Cifar10(Dataset):
    def __init__(self, data, transform, seed=None):
        self.data = data
        self.transform = transform
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

        if seed is not None:
            random.seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def visualize(self, idx=0, samples=10, cols=5):
        """Visualize augmentations applied to an instance with index idx.
        The top-left image in the output is the original."""

        original, label = self.data[idx]
        original = np.array(original)
        transform = A.Compose([t for t in self.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        rows = samples // cols

        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(6, 3))
        ax.ravel()[0].imshow(original)
        ax.ravel()[0].set_axis_off()
        ax.ravel()[0].set_title('Original ({})'.format(self.classes[label]))

        for i in range(1, samples):
            augmented = self.transform(image=original)["image"]
            ax.ravel()[i].imshow(augmented)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()
