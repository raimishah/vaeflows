import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from scipy.stats import norm

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



def evaluate_vae_model(model, X_tensor):
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, _,_,_= model(X_tensor)
    out_pred = out_pred.cpu().detach().numpy()

    idx = 0
    preds = []
    for i in range(len(out_pred)):
        for j in out_pred[i,0]:
            preds.append(j)

    preds = np.array(preds)
    
    return preds




def evaluate_cvae_model(model, X_tensor, c):
    
    cond_window_size = c.size(2)

    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    output, _,_,_= model(X_tensor, c)
    output = output.detach().numpy()

    idx = 0
    preds = []
    for i in range(len(output)):
        for j in output[i,0, :cond_window_size]:
            preds.append(j)

    preds = np.array(preds)
    
    return preds




'''
returns adjusted predictions (like DONUT method) (all others paper use this I believe also)
'''
def evaluate_adjusted_anomalies(real, scores, thresh):
    pointwise_alerts = np.array([1 if scores[i] > thresh else 0 for i in range(len(scores))])

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


def plot_error_and_anomaly_idxs(real, preds, anomaly_idxs):

    plt.figure(figsize=(50,15))
    plt.plot(real)
    plt.plot(preds)
    for ai in anomaly_idxs:
        plt.plot(ai, 1)
    plt.show()
    
    plt.figure(figsize=(50,15))
    scores = (preds - real[:len(preds)])**2
    for idx,ai in enumerate(anomaly_idxs):
        plt.scatter(ai, scores[ai], color='red')
    plt.plot(scores)
    plt.show()

    return



def VAE_test_evaluation(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs):

    real = np.zeros(len(X_test_data), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    preds = evaluate_vae_model(model, X_test_tensor)

    plot_error_and_anomaly_idxs(X_test_data, preds, anomaly_idxs_test)

    real = real[:len(preds)]

    scores = (preds - X_test_data[:len(preds)])**2

    thresh = np.quantile(scores, .98)

    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)

    #plt.scatter(np.arange(len(real)),real,alpha=.5)
    #plt.scatter(np.arange(len(real)),anomaly_preds,alpha=.5)

    #plt.show()


    #rates:
    precision = precision_score(real, anomaly_preds)
    recall = recall_score(real, anomaly_preds)
    f1 = f1_score(real, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))

    precision, recall, thresholds = precision_recall_curve(real, scores)

    #todo later
    aupr_scores = np.copy(scores)
    aupr = average_precision_score(real,scores)

    print('aupr : ' + str(aupr))

    plt.plot(recall, precision)
    plt.show()

    return





def cVAE_test_evaluation(model, X_test_tensor, X_test_data, X_train_data, cond_test_tensor, cond_window_size, anomaly_idxs):

    real = np.zeros(len(X_test_data), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    preds = evaluate_cvae_model(model, X_test_tensor, cond_test_tensor)

    plot_error_and_anomaly_idxs(X_test_data, preds, anomaly_idxs_test)

    real = real[:len(preds)]

    scores = (preds - X_test_data[:len(preds)])**2

    thresh = np.quantile(scores, .98)

    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)


    #rates:
    precision = precision_score(real, anomaly_preds)
    recall = recall_score(real, anomaly_preds)
    f1 = f1_score(real, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))

    precision, recall, thresholds = precision_recall_curve(real, scores)

    #todo later
    aupr_scores = np.copy(scores)
    aupr = average_precision_score(real,scores)

    print('aupr : ' + str(aupr))

    plt.plot(recall, precision)
    plt.show()

    return






def evaluate_prob_decoder_vae_model(model, X_tensor):
    
    X_tensor = X_tensor.cuda()if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, rec_mu, rec_sigma, _ = model(X_tensor)
    out_pred = out_pred.cpu().detach().numpy()
    print(out_pred.shape, rec_mu.shape, rec_sigma.shape)
    probs = []
    for i in range(rec_mu.shape[0]):
        for j in range(rec_mu.shape[2]):
            probs.append(norm.pdf(X_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

    idx = 0
    preds = []
    for i in range(len(out_pred)):
        for j in out_pred[i,0]:
            preds.append(j)
            
    return preds, probs




def prob_decoder_VAE_test_evaluation(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs):

    real = np.zeros(len(X_test_data), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    preds, scores = evaluate_prob_decoder_vae_model(model, X_test_tensor)

    plot_error_and_anomaly_idxs(X_test_data, preds, anomaly_idxs_test)

    real = real[:len(preds)]

    thresh = np.quantile(scores, .98)

    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)

    #plt.scatter(np.arange(len(real)),real,alpha=.5)
    #plt.scatter(np.arange(len(real)),anomaly_preds,alpha=.5)

    #plt.show()

    #rates:
    precision = precision_score(real, anomaly_preds)
    recall = recall_score(real, anomaly_preds)
    f1 = f1_score(real, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))

    precision, recall, thresholds = precision_recall_curve(real, scores)

    #todo later
    aupr_scores = np.copy(scores)
    aupr = average_precision_score(real,scores)

    print('aupr : ' + str(aupr))

    plt.plot(recall, precision)
    plt.show()

    return







def evaluate_prob_decoder_cvae_model(model, X_tensor, c, cond_window_size):
    
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, rec_mu, rec_sigma, _ = model(X_tensor, c)
    out_pred = out_pred.cpu().detach().numpy()
    probs = []
    
    for i in range(rec_mu.shape[0]):
        for j in range(0, cond_window_size):
            probs.append(norm.pdf(X_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

    idx = 0
    preds = []
    mu = []
    sigma = []
    
    for i in range(len(out_pred)):
        for j in range(cond_window_size):
            preds.append(out_pred[i,0,j].item())
            mu.append(rec_mu[i,0,j].item())
            sigma.append(rec_sigma[i,0,j].item())

    return preds, probs



def prob_decoder_cVAE_test_evaluation(model, X_test_tensor, X_test_data, X_train_data, cond_test_tensor, cond_window_size, anomaly_idxs):

    real = np.zeros(len(X_test_data), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    preds, scores = evaluate_prob_decoder_cvae_model(model, X_test_tensor, cond_test_tensor, cond_window_size)


    plot_error_and_anomaly_idxs(X_test_data, preds, anomaly_idxs_test)

    real = real[:len(preds)]

    thresh = np.quantile(scores, .95)

    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)


    #rates:
    precision = precision_score(real, anomaly_preds)
    recall = recall_score(real, anomaly_preds)
    f1 = f1_score(real, anomaly_preds)
    print('precision : ' + str(precision) + ' recall : ' + str(recall) + ' f1 : ' + str(f1))

    precision, recall, thresholds = precision_recall_curve(real, scores)

    #todo later
    aupr_scores = np.copy(scores)
    aupr = average_precision_score(real,scores)

    print('aupr : ' + str(aupr))

    plt.plot(recall, precision)
    plt.show()