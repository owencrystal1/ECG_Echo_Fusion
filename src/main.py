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
    'learning_rate': 1e-7,
    'centerCrop': 0,
    'optim_metric': 'f1',
    'loss_fx': 'ce',
    'weighted_sample': True,
    'weight_decay': 0.3,  # changed from 0.05 -> 0.1 -> 0.3
    'train_from_3channel': False,
    'ecg_fusion_type': 'ML', # CNN or ML
    'focused_leads': ['II', 'V1']
}

if __name__ == "__main__":
    """
    ex: python ecg_train.py
    """
    test_num = 1 # [1, 2, 3, 4, 5, 6, 7]
    cuda_num = 3
    view = 'AP4' # [None, 'AP2', 'AP3', 'AP4', 'PLAX', 'PSAX_M', 'PSAX_V']
    
    # reading command line inputs
    base_path = os.getcwd()
    if base_path[-1] != '/':
        base_path = base_path + r'/'
    params['base_path'] = base_path
    params['save_path'] = './t{}_outputs/'.format(test_num)
    params['dataframe_pkl'] = 'datasplits_fusion.pkl'
    params['ecg_dir'] = './ECG/'
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
            model_ft_resnext101, input_size = model_base.initialize_model(params)

        

        # update parameters with the input_size of the model
        params['input_size'] = input_size

        # Detect if we have a GPU available
        device = torch.device("cuda:{}".format(cuda_num) if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

        df_ecg = pd.read_csv('./data/ECG_features_II_V1_norm.csv')
        
        # load the dataloaders into the dictionary in the form: {'train': train_loader, 'val': test_loader}
        dataloaders_dict = dataloader_base.ecg_dataloader(params, df_ecg, mode='train')

        ap4_weights = torch.load('./models/resnext101_ap4_best.pth') # use an input into training module model_ft.load_state_dict(ap4_weights)


        if params['ecg_fusion_type'] == 'ML':
            model = model_base.ECG_ML_Echo_CNN_Fusion() # we need to load the model - initialize it (line 84) then load it (line 102)
        elif params['ecg_fusion_type'] == 'CNN':
            model = model_base.ECG_CNN_Echo_CNN_Fusion()

        

        # load the model, optimizer, and loss functions
        model_ft, optimizer_ft, criterion = training_base.load_model(params, device, model)


        
        print('model loaded, training now')

        # train and evaluate
        # model_ft, acc_hist, auc_hist = training_base.train_epochs(params, device, model_ft, dataloaders_dict, criterion, optimizer_ft, patience=10)
        
        model_ft = training_base.new_train_epochs(params, device, model_ft, dataloaders_dict, criterion, optimizer_ft, patience=10)

        # save the final versions
        torch.save(model_ft, params['save_path'] + '{}_fusion_best.pth'.format(params['model_name']))

        ###### TESTING #########

        dataloaders_dict = dataloader_base.ecg_dataloader(params, df_ecg, mode='test')


        model_ft = torch.load(params['save_path'] + '{}_fusion_best.pth'.format(params['model_name']))


        pt_ids, true_labels, pred_labels, score_0, score_1, score_2 = training_base.model_predict(device, model_ft, dataloaders_dict)

        results = pd.DataFrame(columns=['pt_id', 'true', 'pred', 'score_0', 'score_1', 'score_2'])
        results.pt_id = pt_ids
        results.true = true_labels
        results.pred = pred_labels
        results.score_0 = score_0
        results.score_1 = score_1
        results.score_2 = score_2
        results.to_csv(params['save_path'] + 'slice_wise_predictions.csv')

