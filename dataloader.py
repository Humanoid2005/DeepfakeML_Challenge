import copy
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

class DeepfakeDataset(Dataset):
    def __init__(self, dir, isTrain, csv_path=None, transform=None, two_views=False):
        self.dir = dir
        self.isTrain = isTrain
        self.transform = transform
        self.two_views = two_views  # return two independently augmented views, stacked as [2, C, H, W]

        split_dir = os.path.join(dir, "train" if isTrain else "test")
        self.img_dir = split_dir if os.path.isdir(split_dir) else dir

        if csv_path is None:
            csv_path = "../ground_truth.csv" if isTrain else "../sample_submission.csv"
        full_csv_path = os.path.join(dir, csv_path) if csv_path else ""  # csv_path="": just list the images in the folder

        if os.path.exists(full_csv_path):
            self.labels = pd.read_csv(full_csv_path)
        else:
            image_ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(self.img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            self.labels = pd.DataFrame({"image_id": image_ids, "label": 0})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform is not None:
            if self.two_views:
                image = torch.stack([self.transform(image), self.transform(image)])
            else:
                image = self.transform(image)

        return image, label

    @staticmethod
    def load_dataloader(dataset, batch_size, val_split=0.0, num_workers=0, shuffle=True, seed=42, val_transform=None):
        if val_split and val_split > 0:
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            generator = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
            if val_transform is not None:
                # Validate on un-augmented images: same indices, eval transform
                val_dataset = copy.copy(dataset)
                val_dataset.transform = val_transform
                val_dataset.two_views = False
                val_subset = Subset(val_dataset, val_subset.indices)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return train_loader, val_loader

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
