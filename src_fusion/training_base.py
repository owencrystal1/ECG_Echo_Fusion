"""Functions where all the code for training is located

TODO: load_model() needs to have criterion and loss function selection
"""
import copy
import time
import pickle
import numpy as np
import torch
import performance_metrics
import torch.nn.functional as F
import torchvision
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CyclicLR

def train_epochs(param, device, model, dataloaders, criterion, optimizer, patience):

    train_losses = []
    val_losses = []
    num_epochs = 100

    early_stopping_patience = patience
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    if param['OvR']:
        criterion = WeightedOvRLoss()
        print('One vs. Rest classification selected.')
    
    if param['scheduler'] == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4, step_size_up=1300)

    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()  # Set model to training model
        running_train_loss = 0.0
    

        # loop through enumaretad batches
        phase = 'train'

        train_labels = []
        train_preds = []
    
        for inputs, labels, IDs, ecg_features in tqdm(dataloaders[phase]):


            # Create a grid of images
            # grid = torchvision.utils.make_grid(inputs, nrow=8, normalize=True, pad_value=0)
            
            # # Convert the grid to a format suitable for matplotlib
            # grid = grid.permute(1, 2, 0).numpy()  # Convert CHW to HWC
            
            # # Set up the matplotlib figure
            # plt.figure(figsize=(12, 12))
            # plt.imshow(grid)
            # plt.axis('off')  # Hide axes
            
            # # Calculate label positions
            # num_images = len(labels)
            # img_height, img_width = grid.shape[:2]
            # grid_width = img_width // 8
            # grid_height = img_height
            
            # for i, label in enumerate(labels):
            #     row = i // 8
            #     col = i % 8
                
            #     x = col * grid_width + grid_width // 2
            #     y = row * grid_height // (num_images // 8) + grid_height // (num_images // 8) // 2
                
            #     plt.text(x, y, f'Label: {label.item()}', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
            
            # plt.tight_layout()
            # plt.savefig('/home/owen/Datacenter_storage/Owen/ECG_Project/src_fusion/batch_images.png')
            # exit()

            

            inputs = inputs.to(device)
            labels = labels.to(device)
            ecg_features = ecg_features.to(device)

            with torch.set_grad_enabled(True):
            
                # Forward pass
                outputs = model(inputs, ecg_features)

                softmax = nn.Softmax()
                score = softmax(outputs)
                # gets the prediction using the highest probability value
                _, preds = torch.max(softmax(outputs), 1)
            
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_train_loss += loss.item()
            train_labels.extend(labels.data.tolist())
            train_preds.extend(preds.tolist())

        epoch_train_loss = running_train_loss / len(dataloaders[phase])
        train_losses.append(epoch_train_loss)
        print('Training Classification Report:', file=param['logger'])
        print(classification_report(train_labels, train_preds), file=param['logger'])

        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        correct = 0
        total = 0

        phase = 'val'
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for inputs, labels, IDs, ecg_features in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                ecg_features = ecg_features.to(device)

                outputs = model(inputs, ecg_features)
                outputs = outputs.to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # counting te number of correct values

                if param['OvR']:
                    val_loss, _ = criterion(outputs, labels, device)
                else:
                    val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                true_labels.extend(labels.data.tolist())
                pred_labels.extend(predicted.tolist())
        
        # Calculate average validation loss and accuracy for the epoch
        epoch_val_loss = running_val_loss / len(dataloaders[phase])
        val_losses.append(epoch_val_loss)
        val_acc = correct / total
        
        # Update the learning rate scheduler
        if param['scheduler']:
            scheduler.step(epoch_val_loss)

        print('Validation Classification Report:', file=param['logger'])
        print(classification_report(true_labels, pred_labels), file=param['logger'])
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}', file=param['logger'])
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
               print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
               break

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(param['save_path'] + 'loss_curves.png')

    model.load_state_dict(best_model_wts)

    return model



def model_predict(device, model, dataloader, params):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    """
    # setting model to evaluate mode
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    score_0 = []
    score_1 = []
    score_2 = []
    score_3 = []
    pt_ids = []
    ovr_probs = []

    for inputs, labels, IDs, ecg_features in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        ecg_features = ecg_features.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs, ecg_features)

            if params['OvR']:
                criterion = WeightedOvRLoss()
                _, score = criterion(outputs, labels, device)

            else:
                softmax = nn.Softmax()
                score = softmax(outputs)
                
            _, preds = torch.max(score, 1)
            score_0_batch = score[:, 0]
            score_1_batch = score[:, 1]
            score_2_batch = score[:, 2]
            score_3_batch = score[:, 3]
            
            pt_ids.extend(IDs)
            true_labels.extend(labels.data.tolist())
            pred_labels.extend(preds.tolist())
            score_0.extend(score_0_batch.tolist())
            score_1.extend(score_1_batch.tolist())
            score_2.extend(score_2_batch.tolist())
            score_3.extend(score_3_batch.tolist())

    results = pd.DataFrame(columns=['pt_id', 'true', 'pred', 'score_0', 'score_1', 'score_2', 'score_3'])
    results.pt_id = pt_ids
    results.true = true_labels
    results.pred = pred_labels
    results.score_0 = score_0
    results.score_1 = score_1
    results.score_2 = score_2
    results.score_3 = score_3
    results.to_csv(params['save_path'] + 'slice_wise_predictions.csv')

    df_avg = results.groupby('pt_id').agg({
        'score_0': 'mean',
        'score_1': 'mean',
        'score_2': 'mean',
        'score_3': 'mean',
        'true': 'mean'
    }).reset_index()

    df_avg['true'] = df_avg['true'].astype(int)

    df_avg.to_csv(params['save_path'] + 'patient_wise_predictions.csv')

    # columns_to_keep = ['score_0', 'score_1', 'score_2', ]

    # new_df = df_avg[columns_to_keep].copy()

    # y_true = df_avg.true.values

    # y_label = label_binarize(y_true.astype(int), classes=[0,1,2])

    # thresholds = performance_metrics.Find_Optimal_Cutoff(y_label, new_df.values, 3)

    # thresh_preds = performance_metrics.generate_metrics(y_true.astype(int), new_df.values, thresholds)

    # performance_metrics.get_performance_metrics(y_true.astype(int), np.array(thresh_preds))
    # performance_metrics.get_metrics(y_true.astype(int), new_df.values,3, ['II', 'V1'], params['save_path'])

    # cm = confusion_matrix(y_true.astype(int),np.array(thresh_preds))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues, colorbar=False)
    # plt.xticks(ticks=[0,1,2],labels=['AMY','HCM','HTN'])
    # plt.yticks(ticks=[0,1,2],labels=['AMY','HCM','HTN'])
    # plt.title('Joint Fusion Model')
    # plt.savefig(params['save_path'] + 'confusion_matrix.png')
    # print('Confusion matrix saved!')

class WeightedOvRLoss(nn.Module):
    def __init__(self):
        super(WeightedOvRLoss, self).__init__()

    def forward(self, inputs, targets, device):
        out1, out2, out3 = torch.sigmoid(inputs).unbind(1) # probability of that class for each 

        batch_out = torch.sigmoid(inputs)

        one_hot_matrix = np.zeros((len(targets), 3), dtype=int)
    
        # Set the appropriate indices to 1 for each label
        for i, label in enumerate(targets):
            one_hot_matrix[i, label] = 1
        one_hot_matrix = torch.tensor(one_hot_matrix, dtype=torch.float)
        one_hot_matrix = one_hot_matrix.to(device)

        loss1 = F.binary_cross_entropy(out1, one_hot_matrix[:,0])
        loss2 = F.binary_cross_entropy(out2, one_hot_matrix[:,1]) 
        loss3 = F.binary_cross_entropy(out3, one_hot_matrix[:,2])

        loss = (loss1 + loss2 + loss3)/3

        return loss, batch_out



def load_model(parameters, device, model_ft):
    model_ft = model_ft.to(device)

    if parameters['load_best']:
        model_ft = torch.load(parameters['best_model'])

    params_to_update = model_ft.parameters()

    #print("Params to learn:")
    if parameters['feature_extract'] == 'True':
        for param in model_ft.parameters():
            param.requires_grad = False
    elif 'partial' in str(parameters['feature_extract']):
        divideBy = int(parameters['feature_extract'].split('_')[-1])
        params_to_update = []
        num_layers = len([name for name, param in model_ft.named_parameters()])
        third = round(num_layers / divideBy)  # a third of the layers
        if third % 2 != 0:  # rounds up to even number for weight and bias combinations
            third += 1
        print('{} layers, freezing the first {} layers'.format(num_layers, third), file=parameters['logger'])
        layer_count = 0
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True and layer_count <= third:
                param.requires_grad = False
            elif param.requires_grad == True and layer_count > third:  # more than a third i.e. freezing 1/3
                params_to_update.append(param)
                print("\t", name)
            layer_count += 1
        print('Freezing 1/{} of the layers'.format(divideBy), file=parameters['logger'])
    #else:
        #print('Learn All Parameters')

    # Observe that all parameters are being optimized
    if parameters['weight_decay'] != 0:
        optimizer_ft = optim.Adam(params_to_update, lr=parameters['learning_rate'],
                                  weight_decay=parameters['weight_decay'])
    else:
        optimizer_ft = optim.Adam(params_to_update)
        #optimizer_ft = optim.Adam(lambda p: p.requires_grad, params_to_update, lr=parameters['learning_rate'])
        

    # Setup the loss fxn
    # be very careful about the loss functions used and the inputs and targets required for each.
    # ex: CrossEntropyLoss() requires input as [batch_size, #classes] and target as 1D numbers
    if parameters['weights'] == []:
        if parameters['loss_fx'] == 'focal':
            criterion = WeightedFocalLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor(parameters['weights']).to(device)
        if parameters['loss_fx'] == 'focal':
            criterion = WeightedFocalLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)

    return model_ft, optimizer_ft, criterion
