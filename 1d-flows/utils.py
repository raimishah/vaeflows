import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor



def plot_train_test_reconstructions(model, X_train_tensor, X_train_data, X_test_tensor,X_test_data):
    torch.no_grad()
    for X_tensor, X_data in [(X_train_tensor,X_train_data),(X_test_tensor,X_test_data)]:
        X_tensor = X_tensor.cuda() if torch.cuda.is_available() else X_tensor.cpu()
        X_tensor.to(device)
        out_pred, _,_,_= model(X_tensor)
        out_pred = out_pred.cpu().detach().numpy()
        
        idx = 0
        preds = []
        for i in range(len(out_pred)):
            for j in out_pred[i,0]:
                preds.append(j)

        plt.figure(figsize=(30,9))
        plt.plot(X_data,label='real')
        plt.plot(preds,label='pred')
        plt.legend()
        plt.show()

        train_squared_error = mean_squared_error(X_data[:len(preds)], preds)
        print('MSE : ' + str(np.round(train_squared_error,3)))



def plot_train_test_reconstructions_cvae(model, X_train_tensor, X_train_data, X_test_tensor, X_test_data, cond_train_tensor, cond_test_tensor, window_size, cond_window_size):
    torch.no_grad()
    
    #train data
    output, _,_,_= model(X_train_tensor, cond_train_tensor)
    output = output.cpu().detach().numpy()
    print(output.shape)

    idx = 0
    preds = []
    for i in range(len(output)):
        for j in output[i,0, :cond_window_size]:
            preds.append(j)
    
    plt.figure(figsize=(30,9))
    plt.plot(X_train_data[cond_window_size:-cond_window_size],label='real')
    plt.plot(preds,label='pred')
    plt.legend()

    plt.show()
    
    train_squared_error = mean_squared_error(X_train_data[cond_window_size:-cond_window_size], preds) * len(preds)

    
    #test data

    output, _,_,_= model(X_test_tensor, cond_test_tensor)
    output = output.detach().numpy()
    print(output.shape)

    idx = 0
    preds = []
    for i in range(len(output)):
        for j in output[i,0, :cond_window_size]:
            preds.append(j)



    plt.figure(figsize=(30,9))
    plt.plot(X_test_data[:-cond_window_size],label='real')
    plt.plot(preds,label='pred')
    plt.legend()

    plt.show()

    
    test_squared_error = mean_squared_error(X_test_data[:-cond_window_size], preds) * len(preds)

    print('train MSE : ' + str(np.round(train_squared_error,3)) + ' test MSE : ' + str(np.round(test_squared_error,3)))
        
        
def plot_train_test_reconstructions_prob_decoder_model(model, X_train_tensor, X_train_data, X_test_tensor,X_test_data):
    for X_tensor, X_data in [(X_train_tensor,X_train_data),(X_test_tensor,X_test_data)]:
        X_tensor = X_tensor.cuda()if torch.cuda.is_available() else X_tensor.cpu()
        X_tensor.to(device)
        out_pred, rec_mu, rec_sigma, _ = model(X_tensor)
        out_pred = out_pred.cpu().detach().numpy()
        probs = []
        for i in range(rec_mu.shape[0]):
            for j in range(rec_mu.shape[2]):
                probs.append(norm.pdf(X_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

        plt.figure(figsize=(20,6))
        plt.plot(probs)
        plt.show()

        idx = 0
        preds = []
        mu = []
        sigma = []
        for i in range(len(out_pred)):
            for j in out_pred[i,0]:
                preds.append(j)
            for j in rec_mu[i,0]:
                mu.append(j.item())
            for j in rec_sigma[i,0]:
                sigma.append(j.item())

        plt.figure(figsize=(20,6))
        plt.plot(X_data,label='real')
        plt.plot(mu,label='rec_mu',alpha=0.5)
        plt.fill_between(np.arange(len(mu)),np.array(mu)-np.exp(np.array(sigma)),np.array(mu)+np.exp(np.array(sigma)),alpha=0.2)
        plt.legend()
        plt.show()
        

        train_squared_error = mean_squared_error(X_data[:len(preds)], preds)
        print('MSE : ' + str(np.round(train_squared_error,3)))





def plot_train_test_reconstructions_prob_decoder_cvae_model(model, X_train_tensor, X_train_data, X_test_tensor, X_test_data, cond_train_tensor, cond_test_tensor, window_size, cond_window_size):
    torch.no_grad()

    X_train_tensor = X_train_tensor.cuda() if torch.cuda.is_available() else X_train_tensor.cpu()
    X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.cuda() if torch.cuda.is_available() else X_test_tensor.cpu()
    X_test_tensor.to(device)
    cond_train_tensor = cond_train_tensor.cuda() if torch.cuda.is_available() else cond_train_tensor.cpu()
    cond_train_tensor.to(device)
    cond_test_tensor = cond_test_tensor.cuda() if torch.cuda.is_available() else cond_test_tensor.cpu()
    cond_test_tensor.to(device)
    
    #train data
    out_pred, rec_mu, rec_sigma, _ = model(X_train_tensor, cond_train_tensor)
    out_pred = out_pred.cpu().detach().numpy()
    probs = []
    
    for i in range(rec_mu.shape[0]):
        for j in range(0, cond_window_size):
            probs.append(norm.pdf(X_train_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

    plt.figure(figsize=(20,6))
    plt.plot(probs)
    plt.show()
    
    idx = 0
    preds = []
    mu = []
    sigma = []
    
    for i in range(len(out_pred)):
        for j in range(cond_window_size):
            preds.append(out_pred[i,0,j].item())
            mu.append(rec_mu[i,0,j].item())
            sigma.append(rec_sigma[i,0,j].item())

    plt.figure(figsize=(20,6))
    plt.plot(X_train_data,label='real')
    plt.plot(mu,label='rec_mu',alpha=0.5)
    plt.fill_between(np.arange(len(mu)),np.array(mu)-np.exp(np.array(sigma)),np.array(mu)+np.exp(np.array(sigma)),alpha=0.2)
    plt.legend()
    plt.show()
    
    train_squared_error = mean_squared_error(X_train_data[cond_window_size:-cond_window_size], preds) * len(preds)

    
    #test data
    out_pred, rec_mu, rec_sigma, _ = model(X_test_tensor, cond_test_tensor)
    out_pred = out_pred.cpu().detach().numpy()
    probs = []
    for i in range(rec_mu.shape[0]):
        for j in range(cond_window_size):
            probs.append(norm.pdf(X_test_tensor[i,0,j].item(),rec_mu[i,0,j].item(),np.exp(rec_sigma[i,0,j].cpu().item())))

    plt.figure(figsize=(20,6))
    plt.plot(probs)
    plt.show()
    
    idx = 0
    preds = []
    mu = []
    sigma = []
    for i in range(len(out_pred)):
        for j in range(cond_window_size):
            preds.append(out_pred[i,0,j].item())
            mu.append(rec_mu[i,0,j].item())
            sigma.append(rec_sigma[i,0,j].item())
        
    plt.figure(figsize=(20,6))
    plt.plot(X_test_data,label='real')
    plt.plot(mu,label='rec_mu',alpha=0.5)
    plt.fill_between(np.arange(len(mu)),np.array(mu)-np.exp(np.array(sigma)),np.array(mu)+np.exp(np.array(sigma)),alpha=0.2)
    plt.legend()
    plt.show()

    test_squared_error = mean_squared_error(X_test_data[:-cond_window_size], preds) * len(preds)

    print('train MSE : ' + str(np.round(train_squared_error,3)) + ' test MSE : ' + str(np.round(test_squared_error,3)))

    






def get_taxi_data_VAE(path, window_size, train_test_split=.5):
    window=window_size
    data = pd.read_csv(path)

    X = np.array(data['value']).reshape(-1,1)

    #normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = np.squeeze(X, 1)

    split_idx = int(len(X) * train_test_split)

    #Make train data
    X_train_data = X[:split_idx]
    X_test_data = X[split_idx:]

    X_train = []
    for i in range(0, X_train_data.shape[0]-window+1, window):
        X_train.append([X_train_data[i:i+window]])

    X_train = np.array(X_train)
    X_train.shape

    X_train = X_train.astype(np.float32)

    X_train_tensor = torch.from_numpy(X_train)
    Y_train_tensor = torch.from_numpy(X_train)

    #Y_train_tensor = torch.from_numpy(Y_train)

    train = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=False)


    #Make test data
    window=window_size

    X_test = []
    for i in range(0, X_test_data.shape[0]-window+1, window):
        X_test.append([X_test_data[i:i+window]])

    X_test = np.array(X_test)
    X_test.shape

    X_test = X_test.astype(np.float32)

    X_test_tensor = torch.from_numpy(X_test)
    Y_test_tensor = torch.from_numpy(X_test)

    #Y_train_tensor = torch.from_numpy(Y_train)

    test = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, X_test_tensor, trainloader, testloader

def get_taxi_data_cVAE(path, window_size, cond_window_size, train_test_split=.5):    
    window = window_size
    conditional_window = cond_window_size

    data = pd.read_csv(path)
    X = np.array(data['value']).reshape(-1,1)

    #normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = np.squeeze(X, 1)

    split_idx = int(len(X) * train_test_split)

    #Make train data
    X_train_data = X[:split_idx]
    X_test_data = X[split_idx:]

    X_train = []
    cond_train = []
    for i in range(0, X_train_data.shape[0]-conditional_window - window + 1, conditional_window):
        X_train.append([X_train_data[i + conditional_window : i + conditional_window + window]])
        cond_train.append([X[i : i + conditional_window]])

    X_train = np.array(X_train)
    cond_train = np.array(cond_train)

    X_train = X_train.astype(np.float32)
    cond_train = cond_train.astype(np.float32)

    X_train_tensor = torch.from_numpy(X_train)
    cond_train_tensor = torch.from_numpy(cond_train)
    Y_train_tensor = torch.from_numpy(X_train)

    #Y_train_tensor = torch.from_numpy(Y_train)

    train = torch.utils.data.TensorDataset(X_train_tensor, cond_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=False)

    #Make test data
    X_test = []
    cond_test = []

    X_test.append([X_test_data[0:window]])
    cond_test.append([X_train_data[-conditional_window:]])

    for i in range(0, X_test_data.shape[0]-conditional_window-window+1, conditional_window):
        X_test.append([X_test_data[i+conditional_window:i+conditional_window+window]])
        cond_test.append([X_test_data[i:i+conditional_window]])

    X_test = np.array(X_test)
    cond_test = np.array(cond_test)

    X_test = X_test.astype(np.float32)
    cond_test = cond_test.astype(np.float32)

    X_test_tensor = torch.from_numpy(X_test)
    cond_test_tensor = torch.from_numpy(cond_test)
    Y_test_tensor = torch.from_numpy(X_test)

    #Y_train_tensor = torch.from_numpy(Y_train)

    test = torch.utils.data.TensorDataset(X_test_tensor, cond_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, trainloader, testloader
