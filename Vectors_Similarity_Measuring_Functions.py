import numpy as np
import re
import pandas as pd

def EuclideanDistance(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    return np.sqrt(np.sum(np.abs(u,v) ** 2))
    
def L2Norm(u):
    return np.sqrt(np.sum(u**2))

def CosineSimilarity(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    UdotV = np.dot(u,v)
    Unorm = L2Norm(u)
    Vnorm = L2Norm(v)
    return UdotV/(Unorm * Vnorm)

def CosineDistance(u,v, normalization = False):
    return 1 - CosineSimilarity(u,v, normalization = normalization)

def MatchingCoeff(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    return np.sum(np.minimum(u,v))

def JaccardDistance(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    return 1 - MatchingCoeff(u,v)/np.sum(np.maximum(u,v))

def DiceDistance(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    return 1 - (2*MatchingCoeff(u,v))/np.sum(u+v)

def Overlap(u,v, normalization = False):
    if normalization:
        u = u/L2Norm(u)
        v = v/L2Norm(v)
    return MatchingCoeff(u,v) / min(np.sum(u), np.sum(v))

def ProbNorm(u):
    return u/np.sum(u)


def ObsOverExp(matrex):
    Observed = matrex
    RowSum = matrex.sum(axis = 1)
    ColSum = matrex.sum(axis = 0)
    AllSum = matrex.sum()
    exp = (RowSum * ColSum)/ AllSum
    return Observed/exp

def PointwiseMutualInfo(matrex):
    return np.log(ObsOverExp(matrex))

def MinEditDis(w1, w2, ins_cost=1, del_cost=1, rep_cost=2):
    w1_length = len(w1)
    w2_length = len(w2)
    w1_w2 = np.zeros((w1_length+1, w2_length+1))
    back_trace = []
    for i in range(w1_length ):
        w1_w2[i+1,0] = w1_w2[i,0] + 1 
    for i in range(w2_length):
        w1_w2[0,i+1] = w1_w2[0,i] + 1 
    for i in range(w1_length):
        for j in range(w2_length):
            insert_cost = w1_w2[i+1,j] + ins_cost
            del_cost = w1_w2[i,j+1] + del_cost
            rep_cost = w1_w2[i,j] if w1[i] == w2[j] else w1_w2[i,j] + rep_cost
            w1_w2[i+1,j+1] = min(insert_cost, del_cost, rep_cost)
    return w1_w2
