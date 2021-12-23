import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
import csv
import cv2
from datasets.dataset import PoIDataset
import numpy as np
  


def _read_csv(filepath):
    images, labels, labels_txt = [], [], []
    width = 224
    height = 224
    # reading csv file
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for (img, label) in csvreader:
            image = cv2.imread(img)
            image_resize = (cv2.resize(image, (width, height)))
            images.append(image_resize)
            labels_txt.append(label)
        labels = (np.unique(labels_txt, return_inverse=True)[1])
    return images, labels

class POIDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 train_batch_size=32,
                 test_batch_size=16,
                 seed=28):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.seed = seed

    def prepare_data(self):
        pass
        
    
    def setup(self, stage="fit"):
        transform = transforms.Compose([transforms.ToTensor()])

        if stage == "fit" or stage is None:
            images, labels = _read_csv(self.train_path)
            x_train, x_val, y_train, y_val =\
                train_test_split(images, labels, test_size=0.3, random_state=self.seed)
            self.train_dataset = PoIDataset(x_train,
                                            y_train, 
                                            transform=transform)
            self.val_dataset = PoIDataset(x_val,
                                          y_val,
                                          transform=transform)

        if stage == "predict" or stage is None:
            images, labels = _read_csv(self.test_path)
            self.test_dataset = PoIDataset(images,
                                           labels, 
                                           transform=transform)

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset,
                              batch_size=self.train_batch_size,
                              shuffle=True, 
                              num_workers=2)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2)

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=2)