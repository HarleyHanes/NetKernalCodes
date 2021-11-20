# -*- coding: utf-8 -*-
#Repository for 

import numpy as np

def GetKernel(kernel_type):
    #Function which returns a kernel function for use in a sklearn SVC algorithm
    # Check Inputs
    if type(kernel_type)!=str:
        raise(Exception("Invalid data type %s for kernel_type" %type(kernel_type)))
    if kernel_type.lower()=="gaussian":
        #kernel_fcn=lambda x1, x2, params: params["sigma"]*np.exp(-(1/2*params["lambda"]**2)*np.sum((x1-x2)**2))
        return GaussKernel

def GaussKernel(x1,x2,params):
    kernel=np.empty((x1.shape[0],x1.shape[0]))
    #VECTORIZE THIS TO IMPROVE SPEED
    for i in range(len(x1)):
        for j in range(len(x2)):
            kernel[i,j] = params["sigma"]*np.exp(-(1/2*params["lambda"]**2)*np.sum((x1[i]-x2[j])**2))
    return kernel
