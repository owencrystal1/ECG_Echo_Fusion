import os
import random
import pickle
import pandas as pd
from PIL import Image

from src.cardiac_echo_processes import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

def makeWeightedSampler(dataset):
    """makes a weighted sampler based on the input dataset"""
    # Compute class weight each class should get the same weight based on its balance
    temp_df = dataset.label
    # get the count of each label
    benCount = temp_df.label[temp_df.label == 0].count()
    malCount = temp_df.label[temp_df.label == 1].count()
    # get weight of each label
    benWeight = 1. / benCount
    malWeight = 1. / malCount
    #print('sliceweights: ben:{}, mal:{}'.format(benWeight, malWeight))
    # make a tensor of the weights
    sample_weights = []
    for label in temp_df.label:
        if label == 0:
            sample_weights.append(benWeight)
        else:
            sample_weights.append(malWeight)
    sample_weights = torch.tensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def ecg_dataloader(params, mode):
    if mode == 'train':
        train_dataset = load_png_wID(pd_file=params['base_path'] + params['dataframe_pkl'], phase='train', view=params['view'],
                           transform=transforms.Compose([
                               transforms.RandomRotation(40),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.Resize((params['input_size'], params['input_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

        test_dataset = load_png_wID(pd_file=params['base_path'] + params['dataframe_pkl'], phase='val', view=params['view'],
                           transform=transforms.Compose([
                               transforms.Resize((params['input_size'], params['input_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))
        if params['weighted_sample']:
            sampler = makeWeightedSampler(train_dataset)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False,
                                                        num_workers=params['workers'], sampler=sampler)
        else:
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                                        num_workers=params['workers'])

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                                                        num_workers=params['workers'])

        return {'train': train_dataloader, 'val': test_dataloader}

    else:
        #, quadrant=params['quadrant']
        dataset = load_png_wID(pd_file=params['base_path'] + params['dataframe_pkl'], phase=mode, view=params['view'],
                           transform=transforms.Compose([
                               transforms.Resize((params['input_size'], params['input_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

        custom_dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                                                   num_workers=params['workers'])

        return custom_dataloader



class load_png(Dataset):
    """Custom HeadCT dataset with just class labels
    note:
        since it's a custom dataset we need to use:
        for i, data in enumerate(train_loader, 0):
        to enumerate over the train loaders, and the i is the ith train_loader and the data is the sample we generate
    """

    def __init__(self, pd_file, phase, quadrant=0, view=None, transform=None):
        """
        Args:
            pd_file (string): Path to the pickled file with image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_pickle(pd_file)
        df = df[phase]
        df.reset_index(inplace=True, drop=True)
        if view != None:
            df_view = df[df['view'] == view]
            df_view.reset_index(inplace=True, drop=True)
            self.label = df_view
        else:
            self.label = df
        self.transform = transform
        self.remove_quadrant = quadrant

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image_path = os.path.join(self.label['path'].iloc[idx])
        image = Image.open(image_path)
        if self.remove_quadrant != 0:
            image_np = crop_external_cardio(np.array(image), quadrant=self.remove_quadrant)
            image = Image.fromarray(image_np)
        image = image.convert('RGB')
        label = self.label['label'].iloc[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = (image, label)

        return sample

class load_png_wID(Dataset):
    """Custom HeadCT dataset with just class labels
    note:
        since it's a custom dataset we need to use:
        for i, data in enumerate(train_loader, 0):
        to enumerate over the train loaders, and the i is the ith train_loader and the data is the sample we generate
    """

    def __init__(self, pd_file, phase, quadrant=0, view=None, transform=None):
        """
        Args:
            pd_file (string): Path to the pickled file with image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_pickle(pd_file)
        if view != None:
            df = df[phase] # only want train part of dict()
            #df.reset_index(inplace=True, drop=True)
            df_view = df[df['view'] == view]
            #df_view.reset_index(inplace=True, drop=True)
            self.label = df_view
        else:
            self.label = df[phase]
            # adding for grogan data for better output
            self.label.label = self.label.label.apply(lambda x: np.abs(x-1))
            print(self.label.label.value_counts())
        self.transform = transform
        self.remove_quadrant = quadrant

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image_path = os.path.join(self.label['path'].iloc[idx])
        image = Image.open(image_path)#.convert('L')  # convert to grayscale
        if self.remove_quadrant != 0:
            image_np = crop_external_cardio(np.array(image), standardized=True, quadrant=self.remove_quadrant)
            image = Image.fromarray(image_np)
        image = image.convert('RGB')
        label = self.label['label'].iloc[idx]
        resid = self.label['pt_id'].iloc[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = (image, label, resid)

        return sample