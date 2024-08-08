"""Functions where all the code for training is located

TODO: load_model() needs to have criterion and loss function selection
"""
import copy
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epochs(param, device, model, dataloaders, criterion, optimizer, patience):
    """Runs the model for the number of epochs given and keeps track of best weights and metrics"""

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    best_loss = 999.0

    val_acc_history = []
    val_auc_history = []
    
    for epoch in range(param['num_epochs']):
        
        print('-' * 10, file=param['logger'])
        print('Epoch {}/{}'.format(epoch + 1, param['num_epochs']), file=param['logger'])
        print('-' * 10, file=param['logger'])

        model, epoch_val_acc, epoch_val_loss, epoch_val_precision, epoch_val_recall, epoch_f1 = train_epoch(
            param, device, model, dataloaders, criterion, optimizer)

        time_elapsed = time.time() - since
        print('Epoch {} complete: {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60), file=param['logger'])

        # deep copy the model if the accuracy and auc improves
        if param['optim_metric'] == 'accuracy':
            if epoch_val_acc > best_acc:
                best_loss = epoch_val_loss
                best_acc = epoch_val_acc
                best_f1 = epoch_f1
                last_improvement_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Improved ACC, Updated weights \n', file=param['logger'])
        elif param['optim_metric'] == 'f1':
            if epoch_f1 > best_f1:
                best_loss = epoch_val_loss
                best_acc = epoch_val_acc
                best_f1 = epoch_f1
                last_improvement_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Improved f1, Updated weights \n', file=param['logger'])
        else:
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_acc = epoch_val_acc
                last_improvement_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Improved loss, Updated weights \n', file=param['logger'])
        param['logger'].flush()
        val_acc_history.append(epoch_val_acc)

        if param['save_every_epoch']:
            model.load_state_dict(best_model_wts)
            torch.save(model, param['save_path'] + '{}.pth'.format(param['model_name']))
            pickle.dump(val_acc_history, open(param['save_path'] + "{}_acc_history.p".format(param['model_name']), "wb"))

            # save plots for the best ones
            # train_acc_hist = [h for h in val_acc_history]
            # plt.title("Validation Accuracy vs. Number of Training Epochs")
            # plt.xlabel("Training Epochs")
            # plt.ylabel("Validation Accuracy")
            # plt.plot(range(1, epoch+2), train_acc_hist)
            # plt.ylim((0, 1.))
            # plt.xticks(np.arange(1, epoch+2, 1.0))
            # plt.savefig(param['save_path'] + '{}_val_acc.png'.format(param['model_name']))
            # # clear figure to generate new one for AUC
            # plt.clf()
        # check if last improved epoch exceeds patience, stop training - early stopping
        if (epoch - last_improvement_epoch) >= patience:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=param['logger'])
    print('Best val Acc: {:4f}'.format(best_acc), file=param['logger'])
    print('Best val AUC: {:4f}'.format(best_auc), file=param['logger'])

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_auc_history


def train_epoch(param, device, model, dataloaders, criterion, optimizer):
    """Train the model for one epoch and calculates different metrics for evaluation
    Current metrics:
        loss - based on the specific loss function
        accuracy - number of correct classifications
        roc_auc_score - to check for classification performance
        precision_score - to check for classification performance
        recall_score - to check for classification performance
    Inputs:
        device: cpu or gpu
        GAN_aug: True/False
        model: neural network
        dataloaders: a dictionary of data loaders with the format = {'train': train_loader, 'val': test_loader}
        criterion: loss function (nn.CrossEntropyLoss(), nn.BCELoss(), etc.)
        optimizer: optimizer (optim.SGD(), optim.adam(), etc.)
    Outputs:
        model - to save the best weights
        epoch_val_acc - to check for improvement
        epoch_val_loss - to check for improvement
        epoch_val_auc - to check for improvement
        epoch_val_precision - to
        epoch_val_recall
        epoch_val_roc_curve
    """

    # Each epoch has a training and validation phase
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        running_total_num = 0
        running_num_true_hem = 0
        running_num_pred_hem = 0
        running_num_true_no_hem = 0
        running_num_pred_no_hem = 0
        true_labels = []
        predict_pos_score = []
        pred_labels = []
        
        batch = 0

        for inputs, labels, IDs in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            print('Phase: {}\tBatch: {}/{} ({:.2%})'.format(phase, batch, len(dataloaders[phase]), (batch/len(dataloaders[phase]))), end='\r')

            batch += 1
            #print(optimizer)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if param['is_inception'] and phase == 'train':
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # softmax used to make the range be [0-1]
                softmax = nn.Softmax()
                score = softmax(outputs)
                # gets the prediction using the highest probability value
                _, preds = torch.max(softmax(outputs), 1)
                # gets the probability value of the positive class (for AUC, precision, recall calculations) only in
                # binary classification
                pos_score = score[:, 1]

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_total_num += preds.shape[0]
            running_num_true_hem += torch.sum(labels.data == 1)
            running_num_pred_hem += torch.sum(preds == 1)
            running_num_true_no_hem += torch.sum(labels.data == 0)
            running_num_pred_no_hem += torch.sum(preds == 0)
            true_labels.extend(labels.data.tolist())
            predict_pos_score.extend(pos_score.tolist())
            pred_labels.extend(preds.tolist())

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_precision = precision_score(true_labels, pred_labels, average='micro')
        epoch_recall = recall_score(true_labels, pred_labels, average='micro')
        epoch_f1 = f1_score(true_labels, pred_labels, average='micro')

        print(classification_report(true_labels, pred_labels), file=param['logger'])
        print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f}  f1: {:.4f}\n'
              .format(phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1), file=param['logger'])
        param['logger'].flush()

        if phase == 'val':
            epoch_val_acc = epoch_acc
            epoch_val_loss = epoch_loss
            epoch_val_precision = epoch_precision
            epoch_val_recall = epoch_recall

    return model, epoch_val_acc, epoch_val_loss, epoch_val_precision, epoch_val_recall, epoch_f1


def model_predict(device, model, dataloader):
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
    pt_ids = []

    for inputs, labels, pt_id in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            softmax = nn.Softmax()
            score = softmax(outputs)
            _, preds = torch.max(softmax(outputs), 1)
            score_0_batch = score[:, 0]
            score_1_batch = score[:, 1]
            score_2_batch = score[:, 2]
            
            pt_ids.extend(pt_id)
            true_labels.extend(labels.data.tolist())
            pred_labels.extend(preds.tolist())
            score_0.extend(score_0_batch.tolist())
            score_1.extend(score_1_batch.tolist())
            score_2.extend(score_2_batch.tolist())

    return pt_ids, true_labels, pred_labels, score_0, score_1, score_2


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weights=None, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.weights = weights
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.weights != None:
            CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        else:
            CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.gamma * CE_loss
        return F_loss.mean()



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
        optimizer_ft = optim.Adam(params_to_update, lr=parameters['learning_rate'])

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
