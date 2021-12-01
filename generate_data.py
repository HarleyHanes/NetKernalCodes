# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 15:36:17 2021

@author: USER
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

def GetSimulatedData(num_samples, models, network_size, sir_params, avg_contacts, infection_cutoff):
    x_adjMat= []
    x_adjVec = []
    x_infVec = []
    x_contactVec = []
    y_time = []
    y_network = []
    num_models = len(models)
    for j in range(num_models):
        network_type = models[j]
        for i in range(int(num_samples/num_models)):
            #Generate Graphs with mean # of contacts equal to avg contacts (except complete)
            if network_type.lower()=="small worlds":
                graph = nx.watts_strogatz_graph(network_size, avg_contacts, .1)
            elif network_type.lower()=="erdos renyi":
                graph = nx.fast_gnp_random_graph(network_size, 2*avg_contacts/(network_size-1))
            elif network_type.lower()=="complete":
                graph = nx.erdos_renyi_graph(network_size, 1)
            elif network_type.lower() == "scale-free":
                graph = nx.barabasi_albert_graph(network_size, avg_contacts)
            # #Draw Adjacency matrix
            ad_mat = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
            # #Simulate infection 
            (ad_mat, infVec, contactVec, time)=SIRnet(sir_params, network_size*infection_cutoff, ad_mat, initInfected=0)
            # #Vectorize
            ad_vec = np.squeeze(ad_mat.flatten())
            # print(ad_mat.shape)
            # print(ad_vec.shape)
            # print(infVec.shape)
            # print(contactVec.shape)
            # print(time.shape)
            # print(network_type.lower())
            # #Save x-Data
            x_adjMat.append(ad_mat)
            x_adjVec.append(ad_vec)
            x_infVec.append(infVec)
            x_contactVec.append(contactVec)
            # #Save y-data
            y_time.append(time)
            y_network.append(network_type.lower())
            
    #Save data
    np.savez("../data/full_data", x_adjMat = x_adjMat, x_adjVec=x_adjVec, x_infVec=x_infVec, 
             x_contactVec= x_contactVec, y_time = y_time, y_network= y_network)
            
        
    
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
        graph = nx.watts_strogatz_graph(network_size,k_smallWorlds,p_smallWorlds)
        #Extract Adjacency Matrix
        ad_mat = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
        #Vectorize
        ad_vec = np.squeeze(ad_mat.flatten())
        #Save Matrix
        x_data.append(ad_vec)
        x_data_2D.append(ad_mat)
        y_data.append(0)
        #y_data.append("small worlds")   
    for i in range(int(num_samples/2)):
        #Generate small-worlds network
        graph = nx.gnp_random_graph(network_size, p_dense)
        #Extract Adjacency Matrix
        ad_mat = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
        #Vectorize
        ad_vec = np.squeeze(ad_mat.flatten())
        #Save Matrix
        x_data.append(ad_vec)
        x_data_2D.append(ad_mat)
        y_data.append(1)
        #y_data.append("dense")
    np.savez("../data/test_data", x_data = x_data, y_data = y_data)
    return (x_data, y_data)

def TrimNetwork(adjMat,infectedNodes,contactDepth,dataLossProb = 0):
    adjMat_trimmed = np.zeros(adjMat.shape)
    for i in range(contactDepth):
        #Get full contacts of infected individual i
        contacts = adjMat[i,:]
        if not dataLossProb==0:
            #Get removed contacts with likelihood
            keptSamples=((np.random.random_sample(size=contacts.shape)+dataLossProb).astype(int)).astype(bool)
            #Remove contacts
            for j in range(len(keptSamples)):
                if not keptSamples[j]:
                    contacts[j]=0
        #Construct trimmed Adjacency
        adjMat_trimmed[i,:]=contacts
        adjMat_trimmed[:,i]=contacts
    return adjMat_trimmed

def SIRnet(thetaVec, N, adjMat, initInfected=0, chainBool=False, plotChain=False):
    '''
    +++++++++++++++++++++++++++++++
    SIRnet function: generates a "fuzzy" random walk of a network from a stochastic
        SIR simulation
    +++++++++++++++++++++++++++++++
    thetaVec     -- parameters of SIR model (see function for specifics)
    N            -- no. of infected at which simulation is terminated
    adjMat       -- adjacency matrix of contact network (should be numpy array)
    initInfected -- index of initial node infected (value between 0 and len(adjMat)-1)
    chainBool    -- Boolean variable controlling if the returned adjacency matrix is
                    the size of the original matrix; if true, it will return only the
                    adjacency matrix of the chain
    plotChain    -- do you want to plot the resulting chain?
                    (will only show the graph if chainBool = True)

    Note that we do not keep track of time because N < len(adjMat)
    '''

    beta = thetaVec[0] # probability of transmission assuming 1 contact
    gamma = thetaVec[1] # recovery rate

    if N > len(adjMat):
        print("ERROR: number of infected must be less than size of adjacency matrix!")
        return

    successBool = False
    # make sure we return subgraph of length w/N nodes
    while not successBool:
        infCount = 1

        # arrays containing total and infected populations
        NVec = np.array(range(len(adjMat))) # total no. of nodes
        infVec = np.array([initInfected], dtype = int)
        recVec = [] # empty array
        time = 0
        timeVec = np.array([0])  #Vector of when outbreak reaches different % infected
        cutoff = .05    #Start saving times at 5% infected

        while (infCount < N) & (infCount > 0):

            # update susceptible pop. array
            deleteVec = np.sort(np.concatenate((infVec,recVec)))
            deleteVec = deleteVec.astype(int)
            susVec = np.delete(NVec, deleteVec)

            # calculate rates of new infections/recoveries
            atRiskSus = np.sum(adjMat[infVec,:], axis=0) # at-risk susceptibles
            atRiskSus = np.where(atRiskSus > 0)
            # take the intersect of at-risk susceptibles and susceptibles
            atRiskSus = np.intersect1d(susVec, atRiskSus)
            rateS2I = beta*len(atRiskSus)

            rateI2R = gamma*len(infVec)

            # Gillespie algorithm but w/o time
            totalRate = rateS2I + rateI2R
            rateS2I = rateS2I / totalRate
            rateI2R = rateI2R / totalRate

            tmpRand = np.random.uniform()
            if tmpRand <= rateS2I:
                # a random susceptible becomes infected
                infNode = np.random.choice(atRiskSus, 1)
                infVec = np.append(infVec, infNode)    #Removed sorting so we can keep
                                                       # track of order of infection
                infCount += 1
            else:
                # an infected individual recovers
                tmpInd = np.random.choice(len(infVec))
                recVec = np.append(recVec, infVec[tmpInd])
                infVec = np.delete(infVec, tmpInd)
                infCount -= 1
                ## end of if statement
            #Compute time of the event
            time += 1/np.sum(totalRate)* np.log(1/tmpRand)
            perecentInfectedRecovered = (infCount + len(recVec))/len(adjMat)
            if perecentInfectedRecovered >= cutoff:
                timeVec = np.append(timeVec, time)
                cutoff +=.05
            
            ## end of while loop

        if infCount == N:
            successBool = True
            contactVec = np.sort(np.concatenate((infVec, recVec)))
            contactVec = contactVec.astype(int)

            # prune the adjacency matrix so it only contains connections to infected nodes
            keepMeMat = np.zeros((len(adjMat),len(adjMat)))
            keepMeMat[contactVec,:] = 1
            keepMeMat[:,contactVec] = 1
            infAdjMat = np.multiply(keepMeMat, adjMat)

            # only useful for plotting
            if plotChain:
                # plot the chain
                colorVec = np.array(['green']*len(infAdjMat))
                colorVec[infVec.astype(int)] = 'red'

            if chainBool:
                # return only the chain
                idxCols = np.argwhere(np.all(infAdjMat[..., :] == 0, axis=0))
                idxRows = np.argwhere(np.all(infAdjMat[..., :] == 0, axis=1))
                infAdjMat = np.delete(infAdjMat, idxCols, axis=1)
                infAdjMat = np.delete(infAdjMat, idxRows, axis=0)
                if plotChain:
                    # plot the chain
                    colorVec = np.delete(colorVec, idxCols)
                    nx.draw_spring(nx.from_numpy_matrix(infAdjMat), node_color=colorVec)
                    plt.show()

            #print("Shape: ", infAdjMat.shape)
            #print("Chain: ", infAdjMat)
        else:
            successBool = False

    ## end of while successBool loop
    return (infAdjMat, infVec, contactVec, timeVec)
    
    

