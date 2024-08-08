# !/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import pickle
import pandas as pd
from src import model_base, training_base, dataloader_base
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
    'weights': [],
    'fusion_model': True,
    'aug_mode': 'trad',
    'num_epochs': 50,
    'learning_rate': 0.000001,
    'centerCrop': 0,
    'optim_metric': 'f1',
    'loss_fx': 'ce',
    'weighted_sample': True,
    'weight_decay': 0.3,  # changed from 0.05 -> 0.1 -> 0.3
    'train_from_3channel': False
}

if __name__ == "__main__":
    """
    ex: python ecg_train.py
    """
    test_num = 14  # [1, 2, 3, 4, 5, 6, 7]
    cuda_num = 3
    view = 'AP4' # [None, 'AP2', 'AP3', 'AP4', 'PLAX', 'PSAX_M', 'PSAX_V']
    
    # reading command line inputs
    base_path = os.getcwd()
    if base_path[-1] != '/':
        base_path = base_path + r'/'
    params['base_path'] = base_path
    params['save_path'] = './t{}_outputs/'.format(test_num)
    params['dataframe_pkl'] = 'missclassified_ECG.pkl'
    params['view'] = view  # None == use all views framewise, any other specific views means just use that view only ['PSAX_V' 'AP2' 'PSAX_M' 'AP3' 'AP4' 'PLAX']
    
    os.makedirs(params['save_path'], exist_ok=True)

    models = ['resnext101']  # ['alexnet', 'vgg11bn', 'resnet50', 'resnext101', 'se-resnext101', 'inceptionresnetv2']
    
    print('setting up logs')
    params['logger'] = open(os.path.join(params['base_path'], 't{}_training_logs_f1_{}.txt'.format(test_num, models[0])), 'a')

    for model in models:
        params['model_name'] = model
        print('Training Parameters:', file=params['logger'])
        print('-' * 10, file=params['logger'])
        print(params, file=params['logger'])
        print('-' * 10, file=params['logger'])
        params['logger'].flush()

        # initialize a model and print it
        if params['train_from_3channel']:
            print('loading 3 channel model to finetune')
            # initialize a 3 channel model
            params['num_classes'] = 3
            model_ft, input_size = model_base.initialize_model(params)
            # load the 3 channel model weights
            model_ft = torch.load('3_channel_model.pth')
            # change the model's last linear layer
            num_ftrs = model_ft.last_linear.in_features
            params['num_classes'] =2
            model_ft.last_linear = torch.nn.Linear(num_ftrs, params['num_classes'])
        else:
            model_ft, input_size = model_base.initialize_model(params)
        # print(model_ft)

        # update parameters with the input_size of the model
        params['input_size'] = input_size

        # Detect if we have a GPU available
        device = torch.device("cuda:{}".format(cuda_num) if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")
        print(device)

        # load the dataloaders into the dictionary in the form: {'train': train_loader, 'val': test_loader}
        dataloaders_dict = dataloader_base.ecg_dataloader(params, mode='test')

        # # load the model, optimizer, and loss functions
        # model_ft, optimizer_ft, criterion = training_base.load_model(params, device, model_ft)
        # print('model loaded, training now')

        # train and evaluate
        model_ft = torch.load(params['save_path'] + '/resnext101_best.pth')

        pt_ids, true_labels, pred_labels, score_0, score_1, score_2 = training_base.model_predict(device, model_ft, dataloaders_dict)

        results = pd.DataFrame(columns=['pt_id', 'true', 'pred', 'score_0', 'score_1', 'score_2'])
        results.pt_id = pt_ids
        results.true = true_labels
        results.pred = pred_labels
        results.score_0 = score_0
        results.score_1 = score_1
        results.score_2 = score_2
        results.to_csv(params['save_path'] + 'results.csv')


