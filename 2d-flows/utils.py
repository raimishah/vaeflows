import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
        preds=np.zeros((out_pred.shape[0]*out_pred.shape[2], out_pred.shape[3]))

        time_idx=0
        
        window_size = X_train_tensor.shape[2]
        
        for i in range(len(out_pred)):
            preds[time_idx:time_idx+window_size, :] = out_pred[i, 0, :window_size, :]
            time_idx += window_size

        
        for i in range(preds.shape[1]):
            plt.figure()
            plt.plot(X_data[:, i],alpha=.5)
            plt.plot(preds[:, i],alpha=.5)
            plt.show()
        
        mse = mean_squared_error(X_data[:len(preds), :], preds)
        print('MSE : ' + str(np.round(mse,5)))
        
        
def plot_train_test_reconstructions_cvae(model, X_train_tensor, X_train_data, X_test_tensor, X_test_data, cond_train_tensor, cond_test_tensor, window_size, cond_window_size):
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
    out_pred, _,_,_= model(X_train_tensor, cond_train_tensor)
    out_pred = out_pred.cpu().detach().numpy()
        
        
    idx = 0
    preds=np.zeros((out_pred.shape[0]*cond_window_size, out_pred.shape[3]))
    
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+cond_window_size, :] = out_pred[i, 0, :cond_window_size, :]
        time_idx += cond_window_size
    
    for i in range(preds.shape[1]):
        plt.figure()
        plt.plot(X_train_data[:, i],alpha=.5)
        plt.plot(preds[:, i],alpha=.5)
        plt.show()

    mse = mean_squared_error(X_train_data[:len(preds), :], preds)
    print('MSE : ' + str(np.round(mse,5)))



    #test data
    out_pred, _,_,_= model(X_test_tensor, cond_test_tensor)
    out_pred = out_pred.cpu().detach().numpy()

    idx = 0
    preds=np.zeros((out_pred.shape[0]*cond_window_size, out_pred.shape[3]))
    
    time_idx=0
    for i in range(len(out_pred)):
        preds[time_idx:time_idx+cond_window_size, :] = out_pred[i, 0, :cond_window_size, :]
        time_idx += cond_window_size
    
    for i in range(preds.shape[1]):
        plt.figure()
        plt.plot(X_test_data[:, i],alpha=.5)
        plt.plot(preds[:, i],alpha=.5)
        plt.show()

    mse = mean_squared_error(X_test_data[:len(preds), :], preds)
    print('MSE : ' + str(np.round(mse,5)))

def read_machine_data(machine_name, window_size, batch_size):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()


    #window it
    window = window_size
    X_train = []
    X_test = []
    
    for i in range(0, X_train_data.shape[0]-window+1, window):
        X_train.append([X_train_data[i:i+window]])

    for i in range(0, X_test_data.shape[0]-window+1, window):
        X_test.append([X_test_data[i:i+window]])
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    
    train = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(X_test_tensor, X_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    print(X_train_tensor.shape, X_test_tensor.shape)
    
    return X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader

    

def read_machine_data_with_validation(machine_name, window_size, batch_size, val_size=.3):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()

    #window it
    window = window_size
    X_train = []
    X_test = []
    
    for i in range(0, X_train_data.shape[0]-window+1, window):
        X_train.append([X_train_data[i:i+window]])

    for i in range(0, X_test_data.shape[0]-window+1, window):
        X_test.append([X_test_data[i:i+window]])
        
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train, X_val = train_test_split(X_train, test_size=val_size, shuffle=False)

    X_train_tensor = torch.from_numpy(X_train)
    X_val_tensor = torch.from_numpy(X_val)
    X_test_tensor = torch.from_numpy(X_test)
    
    train = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    val = torch.utils.data.TensorDataset(X_val_tensor, X_val_tensor)
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    test = torch.utils.data.TensorDataset(X_test_tensor, X_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    print(X_train_tensor.shape, X_test_tensor.shape)
    
    return X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, valloader, testloader





    
def read_machine_data_cvae(machine_name, window_size, cond_window_size, batch_size):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()


    #train data first
    window_size = window_size
    X_train = []
    cond_train = []

    for i in range(0, X_train_data.shape[0]-cond_window_size-window_size+1, cond_window_size):
        X_train.append([X_train_data[i + cond_window_size : i + cond_window_size + window_size]])
        cond_train.append([X_train_data[i : i + cond_window_size]])
        
    X_train = np.array(X_train)
    cond_train = np.array(cond_train)
    X_train_tensor = torch.from_numpy(X_train)
    cond_train_tensor = torch.from_numpy(cond_train)

    train = torch.utils.data.TensorDataset(X_train_tensor, cond_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
        
    #test data now
    X_test = []
    cond_test = []

    X_test.append([X_test_data[0:window_size]])
    cond_test.append([X_train_data[-cond_window_size:]])
        
    for i in range(0, X_test_data.shape[0]-cond_window_size-window_size+1, cond_window_size):
        X_test.append([X_test_data[i+cond_window_size:i+cond_window_size+window_size]])
        cond_test.append([X_test_data[i:i+cond_window_size]])

    X_test = np.array(X_test)
    cond_test = np.array(cond_test)
    X_test_tensor = torch.from_numpy(X_test)
    cond_test_tensor = torch.from_numpy(cond_test)

    test = torch.utils.data.TensorDataset(X_test_tensor, cond_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader



def read_machine_data_cvae_with_validation(machine_name, window_size, cond_window_size, batch_size, val_size=.3):
    
    X_train = pd.read_pickle(machine_name + '_train.pkl')
    X_test = pd.read_pickle(machine_name + '_test.pkl')
    Y_test = pd.read_pickle(machine_name + '_test_label.pkl')

    df_X_train, df_X_test, df_Y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(Y_test)

    X_train_data = df_X_train.values.copy()
    X_test_data = df_X_test.values.copy()


    #train data first
    window_size = window_size
    X_train = []
    cond_train = []

    for i in range(0, X_train_data.shape[0]-cond_window_size-window_size+1, cond_window_size):
        X_train.append([X_train_data[i + cond_window_size : i + cond_window_size + window_size]])
        cond_train.append([X_train_data[i : i + cond_window_size]])
        
    X_train = np.array(X_train)
    cond_train = np.array(cond_train)

    X_train, X_val, cond_train, cond_val = train_test_split(X_train, cond_train, test_size=val_size, shuffle=False)

    X_train_tensor = torch.from_numpy(X_train)
    cond_train_tensor = torch.from_numpy(cond_train)
    X_val_tensor = torch.from_numpy(X_val)
    cond_val_tensor = torch.from_numpy(cond_val)

    train = torch.utils.data.TensorDataset(X_train_tensor, cond_train_tensor)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

    val = torch.utils.data.TensorDataset(X_val_tensor, cond_val_tensor)
    valloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
        
    #test data now
    X_test = []
    cond_test = []

    X_test.append([X_test_data[0:window_size]])
    cond_test.append([X_train_data[-cond_window_size:]])
        
    for i in range(0, X_test_data.shape[0]-cond_window_size-window_size+1, cond_window_size):
        X_test.append([X_test_data[i+cond_window_size:i+cond_window_size+window_size]])
        cond_test.append([X_test_data[i:i+cond_window_size]])

    X_test = np.array(X_test)
    cond_test = np.array(cond_test)
    X_test_tensor = torch.from_numpy(X_test)
    cond_test_tensor = torch.from_numpy(cond_test)

    test = torch.utils.data.TensorDataset(X_test_tensor, cond_test_tensor)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, valloader, testloader
