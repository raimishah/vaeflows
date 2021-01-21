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
    print('Computing AUPR for {} thresholds ... '.format(len(thresholds)))    
    for idx, th in enumerate(thresholds):
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
    
    
'''
evaluate_vae_model(model, X_tensor):
    inputs: model (network), X_tensor (which tensor to evaluate)
    return: prediction
'''
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


'''
VAE_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs, initial_quantile_thresh):
    inputs: model - (network)
            X_test_tensor -(test tensor to eval for anomaly detection)
            X_test_data - (2d data for easy MSE)
            X_train_data - (just to get length for anomaly indices)
            anomaly_idxs - real anomaly idxs on entire dataset (subtract train len to get for test set)
            initial_quantile_thresh - threshold for anomaly
    return: none
    
    description : plots and computes metrics
'''

def VAE_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs, initial_quantile_thresh):

    #inference
    preds = evaluate_vae_model(model, X_test_tensor)
    
    #create real labels
    real = np.zeros(len(preds), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    scores = -(preds - X_test_data[:len(preds)])**2
    
    compute_AUPR(real, scores)
    
    thresh = np.quantile(scores, initial_quantile_thresh)
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)





def evaluate_prob_decoder_vae_model(model, X_tensor):
    
    X_tensor = X_tensor.cuda()if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    out_pred, rec_mu, rec_sigma, _ = model(X_tensor)
    out_pred = out_pred.cpu().detach().numpy()
    print(out_pred.shape, rec_mu.shape, rec_sigma.shape)
    probs = []
    for i in range(rec_mu.shape[0]):
        for j in range(rec_mu.shape[2]):
            probs.append(norm.logpdf(X_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

    idx = 0
    preds = []
    for i in range(len(out_pred)):
        for j in out_pred[i,0]:
            preds.append(j)
            
    return preds, probs
    

'''
VAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test)
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            scores - score/probabilities of observation of real vals according to model
            preds - prediction values
            X_test_data - for compare with preds
            initial_quantile_thresh - threshold for anomaly
            anomaly_idxs_test - anomaly_idxs
    return: none
    
    description : plots and computes metrics (helper function)
'''
def VAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test):
     
    #initial stuff
    initial_threshold = np.quantile(scores, initial_quantile_thresh)
    print('initial threshold : {}'.format(initial_threshold))
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, initial_threshold)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, initial_threshold)
    print_metrics(real, anomaly_preds)

    pareto = genpareto.fit(scores)
    c, loc, scale = pareto

    γ = c
    β = scale

    # pareto dist fitting method
    q = 10e-3 # desired probability to observe

    N_prime = len(scores)
    N_prime_th = (scores < initial_threshold).sum()
    
    final_threshold = initial_threshold - (β/γ) * ( np.power((q*N_prime)/N_prime_th, -γ) - 1 )
    print('final threshold : {}'.format(final_threshold))
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, final_threshold)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, final_threshold)
    print_metrics(real, anomaly_preds)
    
    
'''
VAE_prob_decoder_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs, initial_quantile_thresh):
    inputs: model - (network)
            X_test_tensor -(test tensor to eval for anomaly detection)
            X_test_data - (2d data for easy MSE)
            X_train_data - (just to get length for anomaly indices)
            anomaly_idxs - real anomaly idxs on entire dataset (subtract train len to get for test set)
            initial_quantile_thresh - threshold for anomaly
    return: none
    
    description : plots and computes metrics
'''
def VAE_prob_decoder_anomaly_detection(model, X_test_tensor, X_test_data, X_train_data, anomaly_idxs, initial_quantile_thresh):
    #inference
    preds, scores = evaluate_prob_decoder_vae_model(model, X_test_tensor)
    
    #create real labels
    real = np.zeros(len(scores), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1
  
    compute_AUPR(real, scores)
    

    VAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test)
    
    return
    
    
    
    
    
    
    
'''
evaluate_vae_model(model, X_tensor, c):
    inputs: model (network)
            X_tensor (which tensor to evaluate)
            c - conditional data
            
    return: prediction
'''

def evaluate_cvae_model(model, X_tensor, c):
    
    cond_window_size = c.size(2)

    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    c = c.cuda() if torch.cuda.is_available() else c.cpu()
    c.to(device)
    
    output, _,_,_= model(X_tensor, c)
    output = output.cpu().detach().numpy()

    idx = 0
    preds = []
    for i in range(len(output)):
        for j in output[i,0, :cond_window_size]:
            preds.append(j)

    preds = np.array(preds)
    
    return preds




'''
cVAE_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor X_train_data, anomaly_idxs, initial_quantile_thresh):
    inputs: model - (network)
            X_test_tensor -(test tensor to eval for anomaly detection)
            X_test_data - (2d data for easy MSE)
            cond_test_tensor - conditional data
            X_train_data - (just to get length for anomaly indices)
            anomaly_idxs - real anomaly idxs on entire dataset (subtract train len to get for test set)
            initial_quantile_thresh - threshold for anomaly
    return: none
    
    description : plots and computes metrics
'''
def cVAE_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor, X_train_data, anomaly_idxs, initial_quantile_thresh):

    cond_window_size = cond_test_tensor.shape[2]
    
    #inference
    preds = evaluate_cvae_model(model, X_test_tensor, cond_test_tensor)
    
    #create real labels
    real = np.zeros(len(preds), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1

    scores = -(preds - X_test_data[:len(preds)])**2
    
    compute_AUPR(real, scores)

    thresh = np.quantile(scores, initial_quantile_thresh)
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, thresh)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, thresh)
    print_metrics(real, anomaly_preds)


    
    
    

def evaluate_prob_decoder_cvae_model(model, X_tensor, c, cond_window_size):
    
    X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
    X_tensor.to(device)
    c = c.cuda() if torch.cuda.is_available() else c.cpu()
    c.to(device)
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




'''
cVAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test)
    inputs: real - array of 0/1 for not anomaly, anomaly, respectively
            scores - score/probabilities of observation of real vals according to model
            preds - prediction values
            X_test_data - for compare with preds
            initial_quantile_thresh - threshold for anomaly
            anomaly_idxs_test - anomaly_idxs
    return: none
    
    description : plots and computes metrics (helper function)
'''
def cVAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test):
     
    #initial stuff
    initial_threshold = np.quantile(scores, initial_quantile_thresh)
    print('initial threshold : {}'.format(initial_threshold))
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, initial_threshold)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, initial_threshold)
    print_metrics(real, anomaly_preds)

    pareto = genpareto.fit(scores)
    c, loc, scale = pareto

    γ = c
    β = scale

    # pareto dist fitting method
    q = 10e-3 # desired probability to observe

    N_prime = len(scores)
    N_prime_th = (scores < initial_threshold).sum()
    
    final_threshold = initial_threshold - (β/γ) * ( np.power((q*N_prime)/N_prime_th, -γ) - 1 )
    print('final threshold : {}'.format(final_threshold))
    plot_error_and_anomaly_idxs(X_test_data, preds, scores, anomaly_idxs_test, final_threshold)
    anomaly_preds = evaluate_adjusted_anomalies(real, scores, final_threshold)
    print_metrics(real, anomaly_preds)


'''
cVAE_prob_decoder_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor, X_train_data, anomaly_idxs, initial_quantile_thresh):
    inputs: model - (network)
            X_test_tensor -(test tensor to eval for anomaly detection)
            X_test_data - (2d data for easy MSE)
            cond_test_tensor - conditional data
            X_train_data - (just to get length for anomaly indices)
            anomaly_idxs - real anomaly idxs on entire dataset (subtract train len to get for test set)
            initial_quantile_thresh - threshold for anomaly
    return: none
    
    description : plots and computes metrics
'''
def cVAE_prob_decoder_anomaly_detection(model, X_test_tensor, X_test_data, cond_test_tensor, X_train_data, anomaly_idxs, initial_quantile_thresh):

    cond_window_size = cond_test_tensor.shape[2]
    
    #inference
    preds, scores = evaluate_prob_decoder_cvae_model(model, X_test_tensor, cond_test_tensor, cond_window_size)
    
    #create real labels
    real = np.zeros(len(scores), dtype=np.int)
    anomaly_idxs_test = anomaly_idxs - len(X_train_data)
    real[anomaly_idxs_test] = 1
  
    compute_AUPR(real, scores)

    cVAE_prob_decoder_helper(real, scores, preds, X_test_data, initial_quantile_thresh, anomaly_idxs_test)
    
    return