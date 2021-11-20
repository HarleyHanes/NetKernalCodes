# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:39:19 2021

@author: USER
"""

#Load Packages
import sys
import numpy as np
import sklearn.svm.SVC as SVC
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

#Load pre-defined function
from Kernels

def main(argv=None):
    #----------------Define Global Parameters---------------------
    data_filename='../Data/networkdata.npz'
    kernel_type='gaussian' #gaussian or spectral
    test_train_split=.7
    #Set Tuning vectors (taken from http://www.stats.ox.ac.uk/~sejdinov/teaching/atsml19/KRR.html)
    sigma_vec=10**(np.arange(1,10,dtype=float)/3)
    lambda_vec= 10**(-np.arange(0,9,dtype=float))
    #Set other meta-parameters
    cv_number=5   # what k-fold cross-validation to use
    #----------------Load Data------------------------------------
    #unpack npz file
    data=np.load(data_filename)
    #seperate adjacency matrices and network type
    x_data=data["trimmed adjacency"]
    y_data=data["network type"]
    #Split into Test/ Train
    split_vec=np.ones((len(y_data)*test_train_split))
    split_vec.append(np.zeros((len(y_data)*(1-test_train_split))))
    #randomize ones and zeros
    split_vec=np.roll(split_vec)
    
    x_train=x_data[split_vec]   
    y_train=y_data[split_vec]
    
    x_test=x_data[1-split_vec]
    y_test=y_data[1-split_vec]
    #------------------Import Kernel---------------------
    kernel_fcn = Kernels.GetKernel(kernel_type) #kernel function with inputs params and x_data
    #------------------Tune/ Train SVM and Kernel-----------------------
    split_accuracies=np.empty(cv_number)
    accuracies=np.empty((len(sigma_vec),len(lambda_vec)))
    params=np.empty(2)
    for iSig in range(len(sigma_vec)):
        params[0]=sigma_vec[iSig]
        for iLam in range(len(lambda_vec)):
            params[1]=lambda_vec[iLam]
            #Define Support Vector Machine
            model = SVC(kernel = lambda x_data: kernel_fcn(x_data,x_data,params))
            #Train with 5-fold cross-validation
            cv_results = cross_validate(model, x_train,y_train, cv=cv_number, return_estimator=True)
            #Test accuracy
            for jCV in range(cv_number):
                split_accuracies[jCV]=cv_results["estimator"][jCV].score(x_test,y_test)
            accuracies[iSig][iLam]=np.mean(split_accuracies)
            
                
    
    
    #--------------------------Plot and save Results--------------------
    #Record optimal solutions
    
    #Plot Accuracy Correlations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #Plot sigma Tunning (Guassian)
    #Plot lambda Tunning (Gaussian)
    #Plot mu Tunning (Gaussian vs. Spectral Weights)
    
    
    
if __name__ == "__main__":
  sys.exit(main())

    

