import os
import random
import pickle
import pandas as pd
import neurokit2 as nk
from PIL import Image
from cardiac_echo_processes import *
from sklearn.preprocessing import MinMaxScaler
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")


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


def ecg_dataloader(params, df_ecg, mode):
    if mode == 'train':

        train_dataset = load_png_wID(params=params, pd_file=params['base_path'] + params['dataframe_pkl'], ecg_data=df_ecg, phase='train', ecg_fusion_type=params['ecg_fusion_type'], view=params['view'],
                           transform=transforms.Compose([
                               transforms.RandomRotation(40),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(),
                               transforms.Resize((params['input_size'], params['input_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

        test_dataset = load_png_wID(params=params, pd_file=params['base_path'] + params['dataframe_pkl'], ecg_data=df_ecg, phase='val', ecg_fusion_type=params['ecg_fusion_type'], view=params['view'],
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
        dataset = load_png_wID(params=params, pd_file=params['base_path'] + params['dataframe_pkl'], ecg_data=df_ecg, phase=mode, ecg_fusion_type=params['ecg_fusion_type'], view=params['view'],
                           transform=transforms.Compose([
                               transforms.Resize((params['input_size'], params['input_size'])),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ]))

        custom_dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                                                   num_workers=params['workers'])

        return custom_dataloader

def load_ecg_signals(focused_leads, file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        nums = data.values()
        all_leads = [data[key] for key in focused_leads]
    return torch.tensor(all_leads, dtype=torch.float32)

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
        image = Image.open(image_path) # is this RGBA 
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

    def __init__(self, params, pd_file, phase, ecg_data, ecg_fusion_type, quadrant=0, view=None, transform=None):
        """
        Args:
            pd_file (string): Path to the pickled file with image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df_ecg = ecg_data # dataframe
        self.ecg_fusion_type = ecg_fusion_type
        self.params = params
        

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
        image = Image.open(image_path).convert('L')  # convert to grayscale

        if self.remove_quadrant != 0:
            image_np = crop_external_cardio(np.array(image), standardized=True, quadrant=self.remove_quadrant)
            image = Image.fromarray(image_np)

        hist_norm = np.array(pickle.load(open('normalized_train_array.pkl', 'rb'))).reshape(1, 195239) # reshape to whatever size the vector is
        image = image.resize((442,442), resample=Image.BILINEAR)

        image = np.array(image)
        matched_image = exposure.match_histograms(image, hist_norm)
        matched_image = Image.fromarray(np.uint8(matched_image))

        image = matched_image.convert('RGB')
        
        label = self.label['label'].iloc[idx]
        resid = self.label['pt_id'].iloc[idx]

        if self.transform is not None:
            image = self.transform(image)

        # find row where Echo_File columnn == resid and diagnosis == label
        # there are no 'R0xx IDs from ecg file 
        if len(resid) < 10 or resid.startswith('Amyloidosis'):
            pt_id = resid.split('_')[1]
            if resid.startswith('NC'):
                pt_id = 'RESID' + pt_id[1:]

            else:
                if pt_id.startswith('R00'):
                    pt_id = 'R' + pt_id[3:]
                elif pt_id.startswith('R0'):
                    pt_id = 'R' + pt_id[2:]
        else:
            pt_id = resid

        
        num_features = 26
        row_ecg = self.df_ecg[(self.df_ecg['Echo_File'] == pt_id) & (self.df_ecg['Diagnosis'] == label)].index
        #ecg_features = self.df_ecg.iloc[row_ecg, -num_features:]


        if self.ecg_fusion_type == 'CNN':
            ecg_file = self.df_ecg['File_Name'][row_ecg]
            ecg_signals = load_ecg_signals(self.params['focused_leads'], ecg_file.values[0])
            ecg_signals = ecg_signals[:,:2500]


            scaled_tensor = torch.empty_like(ecg_signals)

            # filter x using PT function
            for i in range(ecg_signals.shape[0]):
                
                # filtering each individual channel/lead signal
                ch1 = nk.ecg_clean(ecg_signals[i,:], sampling_rate=500, method="pantompkins1985") # pre processing ECG signal
                #ch1 = x[i,:]

                # save channel signal into new tensor to be normalized
                scaled_tensor[i,:] = torch.tensor(ch1)

            # min/max normalization 
            scaler = MinMaxScaler()

            for i in range(ecg_signals.shape[0]):
                lead = scaled_tensor[i,:].reshape(-1,1)

                scaled_lead = scaler.fit_transform(lead)
                scaled_tensor[i,:] = torch.tensor(scaled_lead.flatten(), dtype=torch.float32)
                noise = np.random.normal(0, 0.1, scaled_tensor[i,:].shape)

                # Add noise to the tensor
                scaled_tensor[i,:] = scaled_tensor[i,:] + noise

        if self.ecg_fusion_type == 'ML':
            sample = (image, label, resid, ecg_features.values[0])
        elif self.ecg_fusion_type == 'CNN':

            sample = (image, label, resid, scaled_tensor)
        else:
            sample = (image, label, resid)
        return sample

