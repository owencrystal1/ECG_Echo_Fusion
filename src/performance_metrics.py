# Performance metric functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import json
from sklearn.preprocessing import label_binarize, MinMaxScaler
import warnings
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from scipy import stats
import warnings
import re
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

def get_performance_metrics(y_true, y_pred):
    
    true_labels = y_true
    predicted_classes = y_pred

    # Calculate precision, recall, and f1-score for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, labels=[0, 1, 2])

    # Print results for each class
    for i in range(len(precision)):
        print(f"Class {i}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1-score={f1_score[i]:.3f}")

    # Function to calculate bootstrap confidence interval
    def bootstrap_confidence_interval(true_labels, predicted_classes, metric_func, alpha=0.05, n_bootstrap=1000):
        n_samples = len(true_labels)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_true_labels = true_labels[indices]
            bootstrap_predicted_classes = predicted_classes[indices]
            metric_value = metric_func(bootstrap_true_labels, bootstrap_predicted_classes)
            bootstrap_metrics.append(metric_value)
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        lower_bound = np.percentile(bootstrap_metrics, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
        return metric_func(true_labels, predicted_classes), lower_bound, upper_bound

    # Function to calculate accuracy and its confidence interval
    def calculate_accuracy_and_ci(true_labels, predicted_classes, alpha=0.05, n_bootstrap=1000):
        accuracy = accuracy_score(true_labels, predicted_classes)
        bootstrap_acc = bootstrap_confidence_interval(true_labels, predicted_classes, accuracy_score, alpha, n_bootstrap)
        return accuracy, bootstrap_acc[1], bootstrap_acc[2]

    # Calculate accuracy and its confidence interval
    accuracy, acc_lower, acc_upper = calculate_accuracy_and_ci(true_labels, predicted_classes)

    # Print accuracy and confidence interval
    print(f"\nAccuracy: {accuracy:.3f} ({acc_lower:.3f}, {acc_upper:.3f})")

    # Bootstrap confidence intervals for precision, recall, and f1-score for each class
    bootstrap_precision = []
    bootstrap_recall = []
    bootstrap_f1_score = []

    for i in range(len(precision)):
        precision_metric, precision_lower, precision_upper = bootstrap_confidence_interval(
            (true_labels == i), (predicted_classes == i), lambda x, y: precision_recall_fscore_support(x, y, labels=[0, 1], average='binary')[0]
        )
        recall_metric, recall_lower, recall_upper = bootstrap_confidence_interval(
            (true_labels == i), (predicted_classes == i), lambda x, y: precision_recall_fscore_support(x, y, labels=[0, 1], average='binary')[1]
        )
        f1_metric, f1_lower, f1_upper = bootstrap_confidence_interval(
            (true_labels == i), (predicted_classes == i), lambda x, y: precision_recall_fscore_support(x, y, labels=[0, 1], average='binary')[2]
        )
        
        bootstrap_precision.append((precision_metric, precision_lower, precision_upper))
        bootstrap_recall.append((recall_metric, recall_lower, recall_upper))
        bootstrap_f1_score.append((f1_metric, f1_lower, f1_upper))

    # Print confidence intervals for each class
    for i in range(len(precision)):
        print(f"\nClass {i}:")
        print(f"  Precision: {bootstrap_precision[i][0]:.3f} ({bootstrap_precision[i][1]:.3f}, {bootstrap_precision[i][2]:.3f})")
        print(f"  Recall: {bootstrap_recall[i][0]:.3f} ({bootstrap_recall[i][1]:.3f}, {bootstrap_recall[i][2]:.3f})")
        print(f"  F1-score: {bootstrap_f1_score[i][0]:.3f} ({bootstrap_f1_score[i][1]:.3f}, {bootstrap_f1_score[i][2]:.3f})")

def bootstrap_confidence_interval(true_labels, predicted_classes, metric_func, alpha=0.05, n_bootstrap=1000):
    n_samples = len(true_labels)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_true_labels = true_labels[indices]
        bootstrap_predicted_classes = predicted_classes[indices]
        metric_value = metric_func(bootstrap_true_labels, bootstrap_predicted_classes)
        bootstrap_metrics.append(metric_value)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    lower_bound = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    return metric_func(true_labels, predicted_classes), lower_bound, upper_bound

# Function to calculate accuracy and its confidence interval
def calculate_accuracy_and_ci(true_labels, predicted_classes, alpha=0.05, n_bootstrap=1000):
    accuracy = accuracy_score(true_labels, predicted_classes)
    bootstrap_acc = bootstrap_confidence_interval(true_labels, predicted_classes, accuracy_score, alpha, n_bootstrap)

    return accuracy, bootstrap_acc[1], bootstrap_acc[2]


def compute_roc_curves_with_ci(y_true, y_score, num_classes, num_bootstraps=1000, alpha=0.05):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_lower = dict()
    auc_upper = dict()


    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Bootstrap resampling to estimate confidence intervals for AUC
        auc_samples = []
        for _ in range(num_bootstraps):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            bootstrap_true = y_true[indices, i]
            bootstrap_score = y_score[indices, i]
            bootstrap_auc = auc(*roc_curve(bootstrap_true, bootstrap_score)[:2])
            auc_samples.append(bootstrap_auc)

        auc_samples_sorted = np.sort(auc_samples)
        lower_index = int((alpha/2) * num_bootstraps)
        upper_index = int((1 - alpha/2) * num_bootstraps)
        auc_lower[i] = auc_samples_sorted[lower_index]
        auc_upper[i] = auc_samples_sorted[upper_index]

    return fpr, tpr, roc_auc, auc_lower, auc_upper


def get_CIs(y_test_ovr, y_prob_lr, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_ovr[:, i], y_prob_lr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    n_bootstraps = 1000
    rng_seed = 42

    bootstrapped_scores = {i: [] for i in range(n_classes)}

    #rng = np.random.RandomState(rng_seed)

    def interpolate_roc(fpr, tpr, n_points=100):
        fpr_interp = np.linspace(0, 1, n_points)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        return tpr_interp

    for i in range(n_bootstraps):
        rng = np.random.RandomState(rng_seed+i)

        indices = rng.randint(0, len(y_test_ovr), len(y_test_ovr))
        

        if len(np.unique(y_test_ovr[indices], axis=0)) < n_classes:
            continue

        y_test_bootstrap = y_test_ovr[indices]
        y_score_bootstrap = y_prob_lr[indices]

        for j in range(n_classes):
            if len(np.unique(y_test_bootstrap[:,j])) < 2:
                continue

            fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_test_bootstrap[:,j], y_score_bootstrap[:,j])
            bootstrapped_scores[j].append(interpolate_roc(fpr_bootstrap, tpr_bootstrap))

        mean_tpr = dict()
        std_tpr = dict()
        tpr_upper = dict()
        tpr_lower = dict()
        fpr_interp = np.linspace(0,1,100)

        for i in range(n_classes):
            all_tpr = np.array(bootstrapped_scores[i])
            mean_tpr[i] = np.mean(all_tpr, axis=0)
            std_tpr[i] = np.std(all_tpr, axis=0)
            tpr_upper[i] = np.minimum(mean_tpr[i] + 1.96 * std_tpr[i], 1)
            tpr_lower[i] = np.maximum(mean_tpr[i] - 1.96 * std_tpr[i], 0)
        
    return fpr_interp, tpr_lower, tpr_upper

# Define a function to plot ROC curves with confidence intervals for each class
def plot_roc_curves_with_ci_bounds(fpr, tpr, roc_auc, auc_lower, auc_upper, num_classes, fpr_interp, tpr_lower, tpr_upper, leads, save_path):
    plt.figure()
    colors = ['blue', 'red', 'green']  # Specify colors for each class
    classes = ['Amyloidosis', 'HCM', 'HTN']
    for i in range(num_classes):
        # Plot ROC curve
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{classes[i]} (AUC = {roc_auc[i]:.2f} [{auc_lower[i]:.2f}, {auc_upper[i]:.2f}])')

        # Plot confidence intervals as bounds
        plt.fill_between(fpr_interp, tpr_lower[i], tpr_upper[i], color=colors[i], alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.savefig(save_path + './ROC_curves.png')


def get_metrics(y_test, y_prob, num_classes, leads, save_path):

    y_prob_lr = y_prob
    y_test_ovr = label_binarize(y_test, classes=[0,1,2])


    fpr_interp, tpr_lower, tpr_upper = get_CIs(y_test_ovr, y_prob_lr, num_classes)

    fpr, tpr, roc_auc, auc_lower, auc_upper = compute_roc_curves_with_ci(y_test_ovr, y_prob_lr, num_classes=3)

    # Plot ROC curves with upper and lower bounds of confidence intervals for each class
    plot_roc_curves_with_ci_bounds(fpr, tpr, roc_auc, auc_lower, auc_upper, 3, fpr_interp, tpr_lower, tpr_upper, leads, save_path)
    print('ROC curves saved!')

def Find_Optimal_Cutoff(target, predicted, n_classes):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    # must loop through each of the 
    thresholds = []
    fpr = dict()
    tpr = dict()
    threshold = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(target[:, i], predicted[:, i])
        j = np.arange(len(tpr[i]))
        roc = pd.DataFrame({'tf' : pd.Series(tpr[i]-(1-fpr[i]), index=j), 'threshold' : pd.Series(threshold[i], index=j)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        thresholds.append(roc_t['threshold'])

        float_values = [series.iloc[0] for series in thresholds]
    return float_values


def generate_metrics(y_true, probabilities, thresholds):
    preds = []
    for i in range(len(y_true)): # j represents the clas, i represents observation
        
        label = y_true[i] # 0, 1, 2

        threshold = thresholds[label] # decide which threshold

        if probabilities[i, label] >= threshold:
            prediction = label
        else:
            prediction = np.argmax(probabilities[i])
        
        preds.append(prediction)

    return preds