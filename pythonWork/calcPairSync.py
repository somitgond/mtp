# Author: Somit Gond
# calculate pairwise synchronization rate

import os
import numpy as np

# change directory
os.chdir("tcp-dumbbell")

np_file_name = "pairwise_sync_rate"

# number of nodes
nodes = 60

tau = 50

# with sampling rate of 0.1, 10000 entries will be there in between 1000 and 2000
final_mat = np.zeros(shape=(10000, nodes, nodes))

for i in range(0, nodes):
    for j in range(0, nodes):
        if(i == j):
            continue
        file_1 = f"dumbbell-{i+2}.cwnd"
        file_2 = f"dumbbell-{j+2}.cwnd"
        data_1 = np.genfromtxt(file_1, delimiter=8).reshape(-1, 2)
        data_2 = np.genfromtxt(file_2, delimiter=8).reshape(-1, 2)

        # after 1000 seconds
        data_1 = data_1[9844:]
        data_2 = data_2[9844:]
        
        if(len(data_1) != len(data_2)):
            print("data length not equal")
            break
            
        # convert into loss events
        sz = len(data_2)
        # hold loss events
        loss_1 = np.zeros(sz)
        loss_2 = np.zeros(sz)
        
        for t in range(1, sz-1):
            if(data_1[t][1] < data_1[t-1][1]):
                loss_1[t] = 1
            if(data_2[t][1] < data_2[t-1][1]):
                loss_2[t] = 1

        # total readings to do
        total = sz-1
        
        # calculate s_ij
        for k in range(tau, total-tau):
            ni = 0
            nj = 0
            nij = 0
            for l in range(k-tau, k+tau):
                if(loss_1[l] == 0 and loss_2[l] == 0):
                    continue
                if(loss_1[l] == 1 and loss_2[l] == 1):
                    nij += 1
                    ni += 1
                    nj += 1
                elif(loss_1[l] == 1 and loss_2[l] != 1):
                    ni += 1
                elif(loss_2[l] == 1 and loss_1[l] != 1):
                    nj += 1
            if(ni != 0 and nj != 0):
                final_mat[k][i][j] = max(nij/ni, nij/nj)
            elif(ni == 0 and nj != 0):
                final_mat[k][i][j] = nij/nj
            elif(ni != 0 and nj == 0):
                final_mat[k][i][j] = nij/ni

np.save(np_file_name, final_mat)

# load the numpy file with 'np.load(np_file_name)'
