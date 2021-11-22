# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:39:19 2021

@author: USER
"""

#Load Packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import kernels

#Load Functions
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



def main(argv=None):
    #----------------Define Global Parameters---------------------
    data_filename='../data/test_data.npz'
    kernel_type='gaussian' #gaussian or spectral
    test_train_split=.3
    #Set Tuning vectors (taken from http://www.stats.ox.ac.uk/~sejdinov/teaching/atsml19/KRR.html)
    #sigma_vec=10**(np.arange(1,10,dtype=float)/3)
    #lambda_vec= 10**(-np.arange(0,9,dtype=float))
    sigma_vec=10**(np.arange(1,3,dtype=float)/3)
    lambda_vec= 10**(-np.arange(0,3,dtype=float))
    #Set other meta-parameters
    cv_number=5   # what k-fold cross-validation to use
    #----------------Load Data------------------------------------
    #unpack npz file
    data=np.load(data_filename)
    #seperate adjacency matrices and network type
    x_data=data["x_data"]
    y_data=data["y_data"]
    #Split into Test/ Train
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_train_split)
    #------------------Import Kernel---------------------
    kernel_fcn = kernels.GetKernel(kernel_type) #kernel function with inputs params and x_data
    #------------------Tune/ Train SVM and Kernel-----------------------
    split_accuracies=np.empty(cv_number)
    accuracies=np.empty((len(sigma_vec),len(lambda_vec)))
    params=np.empty(2)
    print(x_train.shape)
    print(x_train)
    for iSig in range(len(sigma_vec)):
        params[0]=sigma_vec[iSig]
        for iLam in range(len(lambda_vec)):
            params[1]=lambda_vec[iLam]
            #Define Support Vector Machine
            #model = SVC(kernel = 'rbf')
            #Define SVC with precomputed kernel
            model = SVC(kernel= "precomputed", C=1,  random_state=0)
            #Compute kernel matrix
            kernel_train = kernel_fcn(x_train,x_train,params)
            #Fit model
            model.fit(kernel_train,y_train)
            # Compute Test Kernel
            kernel_test=kernel_fcn(x_test,x_train,params)
            #Test accuracy
            accuracies[iSig][iLam]=model.score(kernel_test,y_test)
            #Formulate wiht 5-fold cross-validation - CURRENTLY BUGGED
            #Train with 5-fold cross-validation
            #cv_results = cross_validate(model, kernel_train,y_train, cv=cv_number, return_estimator=True)
            # for jCV in range(cv_number):
            #     print(kernel_test.shape)
            #     split_accuracies[jCV]=cv_results["estimator"][jCV].score(kernel_test,y_test)
            # accuracies[iSig][iLam]=np.mean(split_accuracies)
    print(accuracies)
            
                
    
    
    #--------------------------Plot and save Results--------------------
    #Record optimal solutions
    
    #Plot Accuracy Correlations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #Plot sigma Tunning (Guassian)
    #ax1.scatter()
    #Plot lambda Tunning (Gaussian)
    #Plot mu Tunning (Gaussian vs. Spectral Weights)
    
    
    
if __name__ == "__main__":
  sys.exit(main())

    

