import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


    
class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []
        

        attribute_to_labels = {
            'male,asian': (0, None), 'male,white': (1, None), 'male,black': (2, None),
            'male,others': (3, None), 'nonmale,asian': (4, None), 'nonmale,white': (5, None),
            'nonmale,black': (6, None), 'nonmale,others': (7, None), 'young': (None, 0),
            'middle': (None, 1), 'senior': (None, 2), 'ageothers': (None, 3)
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label = attribute_to_labels[attribute]


        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_path = row['Image Path']
            mylabel = int(row['Target'])
            
            # Depending on the attribute, check the corresponding label
            if intersec_label is not None and int(row['intersec_label']) == intersec_label:
                self.img.append(img_path)
                self.label.append(mylabel)
            elif age_label is not None and int(row['Ground Truth Age']) == age_label:
                self.img.append(img_path)
                self.label.append(mylabel)
        

    def __getitem__(self, index):
        path = self.img[index]
        img = np.array(Image.open(path))
        label = self.label[index]
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)
    



class ImageDataset_Test_df(Dataset):
    def __init__(self, csv_file, attribute, owntransforms,datasetname):
        self.transform = owntransforms
        self.img = []
        self.label = []
        
        # Mapping from attribute strings to (intersec_label, age_label) tuples
        # Note: if an attribute doesn't correspond to an age label, we use None
        if datasetname == 'ff++' or datasetname == 'dfdc':
            attribute_to_labels = {
                'male,asian': (0, None), 'male,white': (1, None), 'male,black': (2, None),
                'male,others': (3, None), 'nonmale,asian': (4, None), 'nonmale,white': (5, None),
                'nonmale,black': (6, None), 'nonmale,others': (7, None), 'young': (None, 0),
                'middle': (None, 1), 'senior': (None, 2), 'ageothers': (None, 3)
            }
        if datasetname == 'dfd' or datasetname == 'celebdf':
            attribute_to_labels = {
                'male,white': (0, None), 'male,black': (1, None), 'male,others': (2, None),
                'nonmale,white': (3, None), 'nonmale,black': (4, None), 'nonmale,others': (5, None),
                'young': (None, 0), 'middle': (None, 1), 'senior': (None, 2), 'ageothers': (None, 3)
            }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label = attribute_to_labels[attribute]

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_path = row['Image Path']
            mylabel = int(row['Target'])
            
            # Depending on the attribute, check the corresponding label
            if intersec_label is not None and int(row['intersec_label']) == intersec_label:
                self.img.append(img_path)
                self.label.append(mylabel)
            elif age_label is not None and int(row['Ground Truth Age']) == age_label:
                self.img.append(img_path)
                self.label.append(mylabel)

    def __getitem__(self, index):
        path = self.img[index]
        img = np.array(Image.open(path))
        label = self.label[index]
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)

