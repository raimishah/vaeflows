import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from scipy import stats
from scipy.stats import norm
from scipy.stats import genpareto


import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchsummary import summary

import torchvision
from torchvision import datasets
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''
plot_error_and_anomaly_idxs(real, preds, scores, anomaly_idxs, thresh):
    inputs: real - real values
            preds - prediction values
            scores - score/probabilities of observation of real vals according to model
            anomaly_idxs - anomaly indices for testing
            thresh - threshold using for anomaly classification
    return: none
    
'''
def plot_error_and_anomaly_idxs(real, preds, scores, anomaly_idxs, thresh):

    plt.figure(figsize=(50,15))
    plt.plot(real)
    plt.plot(preds)
    for ai in anomaly_idxs:
        plt.plot(ai, 1)
    plt.show()
    
    plt.figure(figsize=(50,15))
    for idx,ai in enumerate(anomaly_idxs):
        plt.scatter(ai, scores[ai], color='red')
    plt.axhline(y=thresh, color='red', label='threshold')
    plt.plot(scores)
    plt.show()

    return

'''
evaluate_adjusted_anomalies(real, scores, thresh):
    inputs: real - real values
            scores - score/probabilities of observation of real vals according to model
            thresh - threshold using for anomaly classification
    return: adjusted_alerts (predictions for anomalies based on DONUT method)
    
    description: computes anomaly based on DONUT method
    
'''
def evaluate_adjusted_anomalies(real, scores, thresh):
    pointwise_alerts = np.array([1 if scores[i] < thresh else 0 for i in range(len(scores))])

    anomaly_windows = []
    i = 0
    while i < len(real):
        if real[i] == 1:
            j = i
            while(j < len(real)):
                if real[j] == 0:
                    anomaly_windows.append([i,j])
                    break
                j+=1

                if j == len(real)-1 and real[j] == 1:
                    anomaly_windows.append([i,j+1])
                    break                

            i = j-1

        i+=1

    adjusted_alerts = np.copy(pointwise_alerts)
    for aw in anomaly_windows:
        if pointwise_alerts[aw[0]:aw[1]].any() == 1:
            adjusted_alerts[aw[0]:aw[1]] = 1


    return adjusted_alerts

'''
print_metrics(real, anomaly_preds):
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            anomaly_preds - prediction 0/1 for not anomaly, anomaly, respectively
    return: none
'''
def print_metrics(real, anomaly_preds):
    print('\n--- Metrics ---')
    precision = precision_score(real, anomaly_preds)
    recall = recall_score(real, anomaly_preds)
    f1 = f1_score(real, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))
    print('\n')

    
    
'''
print_metrics(real, scores):
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            scores - anomaly scores
    return: none
    description: plots PR curve and prints AUPR, best f1
'''    
def compute_AUPR(real, scores):
    precision, recall, thresholds = precision_recall_curve(real, scores)
    
    precisions = []
    recalls = []
    
    f1s = []

    threshold_jump = 5
    
    print('Computing AUPR for {} thresholds ... '.format(len(thresholds[::threshold_jump])))    
    #for idx, th in enumerate(thresholds):
    for idx, th in enumerate(thresholds[::threshold_jump]):
        #if idx%1000==0:
        #    print(idx)
        anomaly_preds = evaluate_adjusted_anomalies(real, scores, th)
        precision = precision_score(real, anomaly_preds)
        recall = recall_score(real, anomaly_preds)
        f1 = f1_score(real, anomaly_preds)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
       
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.plot(recalls, precisions)
    plt.show()
    
    print('\n--- AUPR ---')
    print(auc(recalls, precisions))    

    best_f1_idx = np.argmax(f1s)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1_score = f1s[best_f1_idx]
    
    print('Best F1 score : {} at threshold : {} (1-percentile : {})'.format(best_f1_score, best_f1_threshold, 1-best_f1_score))
    print('Corresponding best precision : {}, best recall : {}'.format(precisions[best_f1_idx], recalls[best_f1_idx]))
    
    
    
    
def evaluate_vae_model(model, X_tensor):
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, _,_,_= model(X_tensor)
    out_pred = out_pred.cpu().detach().numpy()
        
    idx = 0
    preds=np.zeros((out_pred.shape[0]*out_pred.shape[2], out_pred.shape[3]))

    window_size = X_tensor.shape[2]
    
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+window_size, :] = out_pred[i, 0, :window_size, :]
        time_idx += window_size

    return preds
    


def VAE_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, df_Y_test, initial_quantile_thresh):

    real = df_Y_test.values
    real = np.reshape(real, (real.shape[0], ))
    anomaly_idxs = np.where(real == 1)[0]
    
    #inference
    preds = evaluate_vae_model(model, X_test_tensor)
    scores = - (np.square(preds - X_test_data[:len(preds)])).mean(axis=1)

    #create real labels
    real = np.zeros(len(scores), dtype=np.int)
    real[anomaly_idxs] = 1
    
    compute_AUPR(real, scores)
    
    thresh = np.quantile(scores, initial_quantile_thresh)
    #plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)

    
    
    

def evaluate_cvae_model(model, X_tensor, c):
    cond_window_size = c.size(2)
    
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    c = c.cuda() if torch.cuda.is_available() else c.cpu()
    c.to(device)

    out_pred, _,_,_= model(X_tensor, c)
    out_pred = out_pred.cpu().detach().numpy()

    preds=np.zeros((out_pred.shape[0]*cond_window_size, out_pred.shape[3]))
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+cond_window_size, :] = out_pred[i, 0, :cond_window_size, :]
        time_idx += cond_window_size
    
    return preds


def cVAE_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor, X_train_data, df_Y_test, initial_quantile_thresh):

    cond_window_size = cond_test_tensor.shape[2]

    #inference
    preds = evaluate_cvae_model(model, X_test_tensor, cond_test_tensor)
    scores = - (np.square(preds - X_test_data[:len(preds)])).mean(axis=1)

    real = df_Y_test.values
    real = np.reshape(real, (real.shape[0], ))
    real = real[:len(preds)]
    anomaly_idxs = np.where(real == 1)[0]
    
    compute_AUPR(real, scores)
    
    thresh = np.quantile(scores, initial_quantile_thresh)
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)