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
from grakel import GraphKernel
from grakel import Graph
import generat_data

#Load Functions
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def main(argv=None):
    #----------------Define Global Parameters---------------------
    data_filename='../data/diffDegree_data.npz'
    kernel_type='gaussian (sparse)' #gaussian or spectral
    test_train_split = .8
    #Set Tuning vectors (taken from http://www.stats.ox.ac.uk/~sejdinov/teaching/atsml19/KRR.html)
    #sigma_vec=10**(np.arange(1,10,dtype=float)/3)
    #lambda_vec= 10**(-np.arange(0,9,dtype=float))
    c_vec = 10**(np.arange(-4,8, dtype=float))
    p_vec = np.linspace(0,.7,8)
    #depth_vec = np.arange(2,6,dtype=int)
    #Set other meta-parameters
    #cv_number=5   # what k-fold cross-validation to use
    #kf = KFold(n_splits=cv_number, random_state=None)
    #----------------Load Data------------------------------------
    #unpack npz file
    data=np.load(data_filename, allow_pickle=True)
    #seperate adjacency matrices and network type
    x_data=data["x_adjMatSparse"]
    y_data=data["y_network"]
    G_data=[]
    print("Data Size: " + str(x_data.shape[0]))
    print("Formulate Graphs")
    for i in range(x_data.shape[0]):
        G_data.append(Graph(x_data[i]))
    #Split into Test/ Train
    G_train, G_test, y_train, y_test = train_test_split(G_data, y_data, test_size = test_train_split)
    #------------------Import Kernel---------------------
    kernel_fcn = kernels.GetKernel(kernel_type) #kernel function with inputs params and x_data
    #------------------Tune/ Train SVM and Kernel-----------------------
    #split_accuracies=np.empty(cv_number)
    accuracies=np.empty((len(c_vec),len(p_vec)))
    accuracy_best=0
    #params=np.empty(2)
    #print(x_train.shape)
    print("Computing kernels")
    kernel_fcn=GraphKernel({"name": "shortest_path", "normalize": True, "with_labels": False})
    # for iP in range(len(p_vec)):
    #     #Trim Data
    #     for i in range(len(G_train)):
    #         G_train_trimmed[i].adjacency_matrix = generate_data.TrimNetwork(G_train.adjacency_matrix)
    #Compute kernel matrix
    kernel_train = kernel_fcn.fit_transform(G_train)
    # Compute Test Kernel
    kernel_test = kernel_fcn.transform(G_test)
    print("Training SVM")
    for iC in range(len(c_vec)):
        #Define SVC with precomputed kernel
        model = SVC(kernel= "precomputed", C=c_vec[iC],  random_state=0)
        #Fit model
        print("Fitting model")
        model.fit(kernel_train,y_train)
        #Test accuracy
        print("Testing model")
        accuracy = model.score(kernel_test,y_test)
        if accuracy >=accuracy_best:
            y_pred = model.predict(kernel_test)
            confmatrix = confusion_matrix(y_pred, y_test, 
                                          labels=["small worlds", "scale-free", "complete", "erdos renyi"])
            confmatrix = confmatrix#/(confmatrix.sum(axis=1)+.001)
        accuracies[iC]=accuracy
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

    

