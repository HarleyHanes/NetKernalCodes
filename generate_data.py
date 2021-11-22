# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 15:36:17 2021

@author: USER
"""
import numpy as np
import networkx

# def GetDummyData(num_samples,network_size):
#     positive_case_key= 2;
#     eposure_case_key = 1;
#     no_data_key = 0;
#     #Define group differences
#     group1_num_contacts = 
#     group1_num_infections = 
#     group2_num_contacts = 
#     group2_num_infections = 
#     #Define Roll Vectors
#     group1_basevec=2*np.ones(group1_num_infections)
#     group1_basevec.append(np.ones(group1_num_contacts))
#     group1_basevec.append(np.zeros(network_size-group1_num_infections-group1_num_contacts))
#     networkx.
    
def GetFullTestData(num_samples, num_contacts, network_size):
    #Set Global params
    k_smallWorlds = num_contacts
    p_smallWorlds = .1
    p_dense = num_contacts/ network_size
    #Intialize Vectors
    x_data = []
    x_data_2D = []
    y_data = []
    #Get small-worlds test
    for i in range(int(num_samples/2)):
        #Generate small-worlds network
        graph = networkx.watts_strogatz_graph(network_size,k_smallWorlds,p_smallWorlds)
        #Extract Adjacency Matrix
        ad_mat = networkx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
        #Vectorize
        ad_vec = np.squeeze(ad_mat.flatten())
        #Save Matrix
        x_data.append(ad_vec)
        x_data_2D.append(ad_mat)
        y_data.append(0)
        #y_data.append("small worlds")   
    for i in range(int(num_samples/2)):
        #Generate small-worlds network
        graph = networkx.gnp_random_graph(network_size, p_dense)
        #Extract Adjacency Matrix
        ad_mat = networkx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
        #Vectorize
        ad_vec = np.squeeze(ad_mat.flatten())
        #Save Matrix
        x_data.append(ad_vec)
        x_data_2D.append(ad_mat)
        y_data.append(1)
        #y_data.append("dense")
    np.savez("../data/test_data", x_data = x_data, y_data = y_data)
    return (x_data, y_data)
    
    

