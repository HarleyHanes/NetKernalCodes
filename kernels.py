# -*- coding: utf-8 -*-
#Repository for 

import numpy as np
import scipy as sp

def GetKernel(kernel_type):
    #Function which returns a kernel function for use in a sklearn SVC algorithm
    # Check Inputs
    if type(kernel_type)!=str:
        raise(Exception("Invalid data type %s for kernel_type" %type(kernel_type)))
    if kernel_type.lower()=="gaussian":
        #kernel_fcn=lambda x1, x2, params: params["sigma"]*np.exp(-(1/2*params["lambda"]**2)*np.sum((x1-x2)**2))
        return GaussKernelUnscaled
    if kernel_type.lower()=="gaussian (sparse)":
        #kernel_fcn=lambda x1, x2, params: params["sigma"]*np.exp(-(1/2*params["lambda"]**2)*np.sum((x1-x2)**2))
        return GaussKernelSparseUnscaled

def GaussKernelVec(x1,x2,params):
    kernel=np.empty((x1.shape[0],x2.shape[0]))
    #VECTORIZE THIS TO IMPROVE SPEED
    for i in range(len(x1)):
        for j in range(len(x2)):
            kernel[i,j] = params[0]*np.exp(-(1/2*params[1]**2)*np.sum((x1[i]-x2[j])**2))
    return kernel

def GaussKernelUnscaled(x1,x2,sigma):
    kernel=np.empty((x1.shape[0],x2.shape[0]))
    for i in range(len(x1)):
        for j in range(len(x2)):
            kernel[i,j] = np.exp(-(1/2*sigma**2)*np.linalg.norm(x1[i]-x2[j], ord=2))
    return kernel

def GaussKernelSparseUnscaled(x1,x2,sigma):
    kernel=np.empty((x1.shape[0],x2.shape[0]))
    for i in range(len(x1)):
        for j in range(len(x2)):
            x=x1[i]-x2[j]
            kernel[i,j] = np.exp(-(1/2*sigma**2)*sp.sparse.linalg.norm(x, ord='fro'))
    return kernel

