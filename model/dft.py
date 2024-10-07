import numpy as np
import math

def my_dft(x,nx,M):
    k = np.linspace(0,M,M+1)
    W = np.exp(-1j*(math.pi/M)*np.outer(k,nx))
    w = (math.pi/M)*k
    X = np.matmul(W,np.transpose(x))

    return X,w