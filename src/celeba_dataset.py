import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle


class CelebaDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        print("loading data...")
        with open(self.file_path, 'rb') as f:
            self.images, labels = pickle.load(f)
        self.labels = []
        for i, label in enumerate(labels):
            temp = []
            temp.append(label.argmax())
            self.labels.append(np.array(temp))
        self.labels = np.array(self.labels)
        print(self.images.shape, self.labels.shape)
        print("data loaded.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = torch.Tensor(image).permute(2, 0, 1)
        label = torch.Tensor(label)
        return image, label

    def show(self):
        for image, label in zip(self.images, self.labels):
            cv2.imshow(f"{label}", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
