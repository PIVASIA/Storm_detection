from torch.utils.data import Dataset


class PoIDataset(Dataset):
    def __init__(self, 
                 img, 
                 labels,
                 transform=None):
        self.labels = labels
        self.img = img
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.img[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label