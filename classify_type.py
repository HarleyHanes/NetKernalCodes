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
from sklearn.metrics import confusion_matrix



def main(argv=None):
    #----------------Define Global Parameters---------------------
    data_filename='../data/noComplete_data.npz'
    kernel_type='gaussian (sparse)' #gaussian or spectral
    test_train_split = .98
    #Set Tuning vectors (taken from http://www.stats.ox.ac.uk/~sejdinov/teaching/atsml19/KRR.html)
    #sigma_vec=10**(np.arange(1,10,dtype=float)/3)
    #lambda_vec= 10**(-np.arange(0,9,dtype=float))
    sigma_vec = 10**(np.arange(1,3,dtype=float)/3)
    lambda_vec = 10**(-np.arange(1, 8, dtype=float))
    c_vec = 10**(np.arange(-4,8, dtype=float))
    #depth_vec = np.arange(2,6,dtype=int)
    #Set other meta-parameters
    #cv_number=5   # what k-fold cross-validation to use
    #----------------Load Data------------------------------------
    #unpack npz file
    data=np.load(data_filename, allow_pickle=True)
    #seperate adjacency matrices and network type
    x_data=data["x_adjMatSparse"]
    y_data=data["y_network"]
    #Split into Test/ Train
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_train_split)
    #------------------Import Kernel---------------------
    kernel_fcn = kernels.GetKernel(kernel_type) #kernel function with inputs params and x_data
    #------------------Tune/ Train SVM and Kernel-----------------------
    #split_accuracies=np.empty(cv_number)
    accuracies=np.empty((len(sigma_vec),len(lambda_vec), len(c_vec)))
    accuracy_best=0
    #params=np.empty(2)
    print(x_train.shape)
    for iSig in range(len(sigma_vec)):
        print("Computing kernels")
        #Compute kernel matrix
        kernel_train_unscaled = kernel_fcn(x_train,x_train,sigma_vec[iSig])
        # Compute Test Kernel
        kernel_test_unscaled = kernel_fcn(x_test[0:200],x_train,sigma_vec[iSig])
        for iLam in range(len(lambda_vec)):
            kernel_train = kernel_train_unscaled*lambda_vec[iLam]
            kernel_test = kernel_test_unscaled*lambda_vec[iLam]
            for iC in range(len(c_vec)):
                #Define SVC with precomputed kernel
                model = SVC(kernel= "precomputed", C=c_vec[iC],  random_state=0)
                #Fit model
                print("Fitting model")
                model.fit(kernel_train,y_train)
                #Test accuracy
                print("Testing model")
                accuracy = model.score(kernel_test,y_test[0:200])
                if accuracy >=accuracy_best:
                    y_pred = model.predict(kernel_test)
                    confmatrix = confusion_matrix(y_pred, y_test[0:200], 
                                                  labels=["small worlds", "scale-free", "complete", "erdos renyi"])
                    confmatrix = confmatrix#/(confmatrix.sum(axis=1)+.001)
                accuracies[iSig][iLam][iC]=accuracy
                #Formulate with 5-fold cross-validation - CURRENTLY BUGGED
                #Train with 5-fold cross-validation
                #cv_results = cross_validate(model, kernel_train,y_train, cv=cv_number, return_estimator=True)
                # for jCV in range(cv_number):
                #     print(kernel_test.shape)
                #     split_accuracies[jCV]=cv_results["estimator"][jCV].score(kernel_test,y_test)
                # accuracies[iSig][iLam]=np.mean(split_accuracies)
    print(accuracies)
    print(y_test[0:20])
    print(y_pred[0:20])
    print(confmatrix)
                
    
    
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

    

