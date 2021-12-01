# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:01:21 2021

@author: USER
"""
import numpy as np
import sys
from generate_data import GetSimulatedData as GetFull

def main(argv=None):
    models= ["small worlds", "erdos renyi", "complete", "scale-free"]
    network_size=200
    sir_params= np.array([0.5, 0.1])
    avg_contacts = 5
    infection_cutoff=.3
    GetFull(4000, models, network_size, sir_params, avg_contacts, infection_cutoff)
    
    data=np.load("../data/full_data.npz", allow_pickle=True)
    
    print(data["x_adjMat"].shape)
    print(data["x_adjVec"].shape)
    print(data["y_network"].shape)
    print(data["y_time"].shape)    
    
    print(data["x_adjMat"][0])
    print(data["x_adjVec"][0])
    print(data["y_network"][0])
    print(data["y_time"][0])
    
if __name__ == "__main__":
  sys.exit(main())