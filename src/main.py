# !/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import pickle
import pandas as pd
import model_base
import training_base
import dataloader_base
import performance_metrics
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


params = {
    'workers': 4,
    'num_classes': 3,
    'ngpu': 3,
    'batch_size': 32,  # changed from 16 -> 32
    'feature_extract': False,
    'load_best': False,
    'save_every_epoch': True,
    'is_inception': False,
    'use_pretrained': True,
    'fusion_model': True,
    'weights':[],
    'aug_mode': 'trad',
    'num_epochs': 50,
    'learning_rate': 1e-7,
    'centerCrop': 0,
    'optim_metric': 'f1',
    'loss_fx': 'ce',
    'weighted_sample': True,
    'weight_decay': 0.3,
    'train_from_3channel': False,
    'ecg_fusion_type': 'ML', # CNN or ML
    'focused_leads': ['II', 'V1'],
    'pretrained': 'AP4', # AP4 or imagenet
    'scheduler': 'cyclic',
    'OvR': True
}

if __name__ == "__main__":
    """
    ex: python ecg_train.py
    """
    test_num = '_ovr2' # [1, 2, 3, 4, 5, 6, 7]
    cuda_num = 0
    view = 'AP4' 
    num_ecg_leads = len(params['focused_leads'])
    num_ecg_features = 13*num_ecg_leads
    
    # reading command line inputs
    base_path = os.getcwd()
    if base_path[-1] != '/':
        base_path = base_path + r'/'
    params['base_path'] = base_path
    params['save_path'] = './t{}_outputs/'.format(test_num)
    params['dataframe_pkl'] = 'datasplits_fusion.pkl'
    params['ecg_dir'] = '/home/owen/Datacenter_storage/ECG_project/Data/HTN_AMY_HCM/ECG/'
    params['view'] = view 
    
    os.makedirs(params['save_path'], exist_ok=True)

    models = ['resnext101']  # ['alexnet', 'vgg11bn', 'resnet50', 'resnext101', 'se-resnext101', 'inceptionresnetv2']
    
    print('setting up logs')
    params['logger'] = open(os.path.join(params['save_path'], 't{}_training_logs_{}.txt'.format(test_num, models[0])), 'a')

    for model in models:
        params['model_name'] = model
        print('Training Parameters:', file=params['logger'])
        print('-' * 10, file=params['logger'])
        print(params, file=params['logger'])
        print('-' * 10, file=params['logger'])
        params['logger'].flush()

        # get input size based on model selected
        _, input_size = model_base.initialize_model(params)

        # update parameters with the input_size of the model
        params['input_size'] = input_size

        # Detect if we have a GPU available
        device = torch.device("cuda:{}".format(cuda_num) if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

        # load and normalize ECG features
        df_ecg = pd.read_csv('./data/ECG_features_II_V1.csv')

        all_features = df_ecg[df_ecg.columns[-num_ecg_features:]]
        meta_data = df_ecg[df_ecg.columns[:-num_ecg_features]]

        scaler = MinMaxScaler(feature_range=(0,1))
        X = pd.DataFrame(scaler.fit_transform(all_features), columns=all_features.columns)

        norm_df = pd.concat([meta_data, X], axis=1)

        # load the dataloaders into the dictionary in the form: {'train': train_loader, 'val': test_loader}
        dataloaders_dict = dataloader_base.ecg_dataloader(params, norm_df, mode='train')

        # define model based on fusion type
        if params['ecg_fusion_type'] == 'ML':
            model = model_base.ECG_ML_Echo_CNN_Fusion(params['pretrained']) 

        elif params['ecg_fusion_type'] == 'CNN':
            model = model_base.ECG_CNN_Echo_CNN_Fusion()
            

        # load the model, optimizer, and loss functions
        model_ft, optimizer_ft, criterion = training_base.load_model(params, device, model)
        
        print('model loaded, training now')


        model_ft = training_base.train_epochs(params, device, model_ft, dataloaders_dict, criterion, optimizer_ft, patience=10)

        # save the final versions
        torch.save(model_ft, params['save_path'] + '{}_fusion_best.pth'.format(params['model_name']))

        ###### TESTING #########

        dataloaders_dict = dataloader_base.ecg_dataloader(params, df_ecg, mode='test')


        model_ft = torch.load(params['save_path'] + '{}_fusion_best.pth'.format(params['model_name']))


        training_base.model_predict(device, model_ft, dataloaders_dict, params)


