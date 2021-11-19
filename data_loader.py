import pandas as pd
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
import random
import pickle



class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.image_dir = image_dir
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info["url"].to_numpy())
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info["label"].to_numpy())

        self.transform = transform

        self.classes = np.unique(self.label_arr).tolist()
        print("******number of classes******", len(self.classes))
        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}

        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        url = self.image_arr[index]
        single_image_name = url.split("/")[-1]
        single_image_name = os.path.join(self.image_dir, single_image_name)
        # Open image
        # print(single_image_name)
        img_as_img = Image.open(single_image_name)
        img_as_img = img_as_img.convert('RGB')
        img_as_tensor = self.transform(img_as_img)

        # print (img_as_tensor.size())

        # # Transform image to tensor
        # img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        label = self.class_to_idx[single_image_label]
        # print (single_image_label)
        # print ("type=", type(single_image_label))
        return img_as_tensor, label

    def __len__(self):
        return self.data_len