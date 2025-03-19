# calculating loss synchronization

import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.chdir("tcp-dumbbell/")

    # reading files
    data_d2 = np.genfromtxt("dumbbell-2.cwnd", delimiter=8).reshape(-1, 2)
    data_d3 = np.genfromtxt("dumbbell-3.cwnd", delimiter=8).reshape(-1, 2)

    # preprocessing data
    data_d2 = data_d2[np.any(data_d2 > 200, axis=1), :]
    data_d3 = data_d3[np.any(data_d3 > 200, axis=1), :]

    # length of d2 and d3 are equal
    print(len(data_d3) == len(data_d3))

    sz = len(data_d2)
    # hold loss events
    loss_d2 = np.zeros(sz)
    loss_d3 = np.zeros(sz)

    for i in range(1, sz):
        if(data_d2[i][1] < data_d2[i-1][1]):
            print("loss d2")
            loss_d2[i] = 1
        if(data_d3[i][1] < data_d3[i-1][1]):
            print("loss d3")
            loss_d3[i] = 1

    # two arrays determining loss events
    s_d2_d3 = np.zeros(len(data_d2))

    total = sz
    tau = 10
    # s_ij = max(n_ij/n_i, n_ji/n_j)

    # sliding window
    for i in range(tau, total-tau):
        ni = 0
        nj = 0
        nij = 0
        for j in range(i-tau, i+tau):
            if(loss_d2[j] == 0 and loss_d3[j] == 0):
                continue
            if(loss_d2[j] == 1 and loss_d3[j] == 1):
                nij += 1
                ni += 1
                nj += 1
            elif(loss_d2[j] == 1 and loss_d3[j] != 1):
                ni += 1
            elif(loss_d3[j] == 1 and loss_d2[j] != 1):
                nj += 1
        if(ni != 0 and nj != 0):
            s_d2_d3[i] = max(nij/ni, nij/nj)
        elif(ni == 0 and nj != 0):
            s_d2_d3[i] = nij/nj
        elif(ni != 0 and nj == 0):
            s_d2_d3[i] = nij/ni

    print(max(s_d2_d3))
