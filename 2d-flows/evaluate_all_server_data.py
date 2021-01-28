import os
import argparse
import glob

import numpy as np

import torch

import utils
import evaluation_utils

def get_scores(model_type, model, X_test_tensor, X_test_data, X_train_data, df_Y_test, cond_test_tensor=None):
    real = df_Y_test.values
    real = np.reshape(real, (real.shape[0], ))
    anomaly_idxs = np.where(real == 1)[0]
    
    #inference
    if model_type=='vae':
        preds = evaluation_utils.evaluate_vae_model(model, X_test_tensor)
    elif model_type=='cvae':
        cond_window_size = cond_test_tensor.shape[2]
        preds = evaluation_utils.evaluate_cvae_model(model, X_test_tensor, cond_test_tensor)

    scores = - (np.square(preds - X_test_data[:len(preds)])).mean(axis=1)

    #create real labels
    real = np.zeros(len(scores), dtype=np.int)
    real[anomaly_idxs] = 1

    return scores, real



def evaluate_models_from_folder(model_type, folder_path, batch_size):

    saved_models = os.listdir(folder_path)

    tn_fp_fn_tp=np.empty((0,4)) 


    for model_file in saved_models:
        print(model_file)
        if '.pth' not in model_file:
            continue
        model = torch.load('{}/{}/{}'.format(os.getcwd(),folder_path,model_file))

        #get real labels
        machine_name = model_file.split('-')
        machine_name = machine_name[1][-1] + '-' + machine_name[2].split('.')[0]

        if model_type=='vae':
            X_train_data, X_test_data, X_train_tensor, X_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data('../../datasets/ServerMachineDataset/machine-' + machine_name, model.window_size, batch_size)
            
            scores,real = get_scores(model_type, model, X_test_tensor, X_test_data, X_train_data, df_Y_test, None)

        if model_type=='cvae':
            X_train_data, X_test_data, X_train_tensor, cond_train_tensor, X_test_tensor, cond_test_tensor, df_Y_test, trainloader, testloader = utils.read_machine_data_cvae('../../datasets/ServerMachineDataset/machine-' + machine_name, model.window_size, model.cond_window_size, batch_size)
            
            scores,real = get_scores(model_type, model, X_test_tensor, X_test_data, X_train_data, df_Y_test, cond_test_tensor)

        confusion_matrix_metrics = evaluation_utils.compute_AUPR(real, scores, threshold_jump=10)
        print(confusion_matrix_metrics)
        tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])

        f = open('{}/{}/{}_confusion_metrics.txt'.format(os.getcwd(),folder_path,machine_name), "w")
        f.write('{} {} {} {}'.format(confusion_matrix_metrics[0,0],confusion_matrix_metrics[0,1],confusion_matrix_metrics[0,2],confusion_matrix_metrics[0,3]))
        f.close()



    tn = tn_fp_fn_tp[:, 0].sum()
    fp = tn_fp_fn_tp[:, 1].sum()
    fn = tn_fp_fn_tp[:, 2].sum()
    tp = tn_fp_fn_tp[:, 3].sum()
    
    print('TN sum : {}'.format(tn))        
    print('FP sum : {}'.format(fp))       
    print('FN sum : {}'.format(fn))        
    print('TP sum : {}'.format(tp))

    F1 = tp / (tp+.5*(fp+fn))
    print('Overall F1 best : {}'.format(F1))        



def get_combined_metrics(save_dir):
    txt_files = os.listdir(save_dir)
    
    tn_fp_fn_tp=np.empty((0,4)) 

    for txt_file in txt_files:
        if 'confusion_metrics.txt' not in txt_file:
            continue
        else:
            f = open('{}/{}/{}'.format(os.getcwd(),save_dir,txt_file), 'r')
            lines = f.readlines()[0]
            confusion_matrix_metrics = np.array([int(i) for i in lines.split(' ')]).reshape(1,4)
            tn_fp_fn_tp = np.concatenate([tn_fp_fn_tp, confusion_matrix_metrics])

    tn = tn_fp_fn_tp[:, 0].sum()
    fp = tn_fp_fn_tp[:, 1].sum()
    fn = tn_fp_fn_tp[:, 2].sum()
    tp = tn_fp_fn_tp[:, 3].sum()
    
    print('TN sum : {}'.format(tn))        
    print('FP sum : {}'.format(fp))       
    print('FN sum : {}'.format(fn))        
    print('TP sum : {}'.format(tp))

    F1 = tp / (tp+.5*(fp+fn))
    print('Overall F1 best : {}'.format(F1)) 


def main():

    model_type='vae'
    save_dir = 'saved_models/regular_vae'

    #evaluate_models_from_folder(model_type, save_dir, 256)

    #get_combined_metrics(save_dir)


if __name__ == '__main__':
    main()
