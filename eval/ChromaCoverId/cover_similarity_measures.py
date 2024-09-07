# -*- coding: utf-8 -*-

"""
Some set of functions for computing audio similarity measures for the task of cover song detection

* cross_recurrent_plots
* qmax_measure [1]
* dmax_measure [2]

[1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification.
     New Journal of Physics.

[2]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification. Multimedia
     Tools and Applications.
     
------
Albin Andrew Correya
@2017
"""

from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from numba import jit


def global_hpcp(chroma):
    """Computes global hpcp of a input chroma vector"""
    if chroma.shape[1] not in [12, 24, 36]:
        raise IOError("Wrong axis for the input chroma array. Use numpy swapaxis for swapping the axis of input array")
    return np.sum(chroma, axis=0) / np.max(np.sum(chroma, axis=0))


def optimal_transposition_index(chromaA, chromaB, n_shifts=12):
    """
    Computes optimal transposition index (OTI) for the chromaB to be transposed in the same key as of chromaA
    Input :
            chromaA : chroma feature array of the query song for reference key
            chromaB : chroma feature array of the reference song for which OTI has to be applied
        Params:
                n_shifts: (default=12) Number of oti tranpositions to be checked for circular shift
    Output : Integer value specifying optimal transposition index for transposing chromaB to chromaA to be in same key
    """
    global_hpcpA = global_hpcp(chromaA)
    global_hpcpB = global_hpcp(chromaB)
    idx = list()
    for index in range(n_shifts):
        idx.append(np.dot(global_hpcpA, np.roll(global_hpcpB, index)))
    return np.argmax(idx)


def transpose_by_oti(chromaB, oti=0):
    """
    Transpose the chromaB vector to a common key by a value of optimal transposition index
    Input :
            chromaB : input chroma array
    Output : chromaB vector transposed to a factor of specified OTI
    """
    return np.roll(chromaB, oti)


def to_timeseries(input_xrray, tau=1, m=9):
    """
    Construct a time series with delay embedding 'tau' and embedding dimension 'm' from the input audio feature vector
    Input :
            input_xrray : input feature array for constructing the timespace embedding
    Params :
            tau (default : 1): delay embedding
            m (default : 9): embedding dimension

    Output : Time series representation of the input audio feature vector
    """

    timeseries = list()
    for startTime in range(0, input_xrray.shape[0] - m*tau, tau):
        stack = list()
        for idx in range(startTime, startTime + m*tau, tau):
            stack.append(input_xrray[idx])
        timeseries.append(np.ndarray.flatten(np.array(stack)))
    return np.array(timeseries)

@jit(nopython=True)
def gamma_state(value, gammaO=0.5, gammaE=0.5):
    """
    This prevents negative entry in Qi,j while calculating Qmax
    Input :
            value : input value
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disruption extension
    Output : gammaO if value is 1 or gammaE if value is 0
    """
    if value == 1.:
        return gammaO
    elif value == 0.:
        return gammaE
    else:
        raise ValueError("Non-binary numbers found in the cross recurrent matrix...")


def cross_recurrent_plot(input_x, input_y, tau=1, m=9, kappa=0.095, transpose=True, swapaxis=False):
    """
    Constructs the Cross Recurrent Plot of two audio feature vector as mentioned in [1]
    Inputs :
            input_x : input feature array of query song
            input_y : input feature array of reference song
    Params :
            kappa (default=0.095)       : fraction of mutual nearest neighbours to consider [0, 1]
            tau (default=1)             : delay embedding [1, inf]
            m (default=9)               : embedding dimension for the time series embedding [0, inf]
            swapaxis (default=False)    : swapaxis of the feature array if it not in the shape (x,12) where x is the \
                                          time axis
            transpose (default=True)    : boolean to check to choose if OTI should be applied to the reference song

    Output : Binary similarity matrix where 1 constitutes a similarity between
            two feature vectors at ith and jth position respectively and 0 denotes non-similarity
    """
    if transpose:
        oti = optimal_transposition_index(input_x, input_y)
        input_y = transpose_by_oti(input_y, oti) #transpose input_y to the key of input_x by a oti value

    if swapaxis:
        input_x = np.swapaxes(input_x,1,0)
        input_y = np.swapaxes(input_y,1,0)

    timespaceA = to_timeseries(input_x, tau=tau, m=m)
    timespaceB = to_timeseries(input_y, tau=tau, m=m)
    pdistances = euclidean_distances(timespaceA, timespaceB)
    transposed_pdistances = pdistances.T
    eph_x = np.percentile(pdistances, kappa*100, axis=1)
    eph_y = np.percentile(transposed_pdistances, kappa*100, axis=1)
    x = eph_x[:,None] - pdistances
    y = eph_y[:,None] - transposed_pdistances
    #apply heaviside function to the array (Binarize the array)
    x = np.piecewise(x, [x<0, x>=0], [0,1])
    y = np.piecewise(y, [y<0, y>=0], [0,1])
    cross_recurrent_plot = x*y.T
    return cross_recurrent_plot

@jit(nopython=True)
def qmax_measure(crp, gammaO=0.5, gammaE=0.5):
    """
    Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]
    Inputs :
            crp : 2-d binary matrix of cross recurrent plot
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disurption extension
    Output : qmax similarity measure from a crp matrix

    [1]. Serra, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover
        song identification. New Journal of Physics, 11.

    """
    Nx = crp.shape[0]
    Ny = crp.shape[1]
    cum_matrix = np.zeros(crp.shape)
    #for the moment doing it in the matlab way
    for i in range(2,Nx):
        for j in range(2,Ny):
            if int(crp[i,j]) == 1:
                cum_matrix[i][j] = max([cum_matrix[i-1][j-1], cum_matrix[i-2][j-1], cum_matrix[i-1][j-2]]) + 1
            else:
                cum_matrix[i][j] = max([0,
                                            (cum_matrix[i-1][j-1] - gamma_state(crp[i-1][j-1], gammaO, gammaE)),
                                            (cum_matrix[i-2][j-1] - gamma_state(crp[i-2][j-1], gammaO, gammaE)),
                                            (cum_matrix[i-1][j-2] - gamma_state(crp[i-1][j-2], gammaO, gammaE))]
                                            )
    #print "Cumulative Matrix computed with shape :", cum_matrix.shape
    if np.max(cum_matrix)==0:
        print("*Cum_matrix max is Zero")
    qmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))
    return qmax, cum_matrix

@jit(nopython=True)
def dmax_measure(crp, gammaO=0.5, gammaE=0.5):
    """
    Computes distance cover song similarity measure from the cross recurrent plots as mentioned in [1]

    Inputs :
            crp : 2-d binary matrix of cross recurrent plot
    Params :
            gammaO : penalty for a disurption onset
            gammaE : penalty for a disurption extension
    Output : dmax similarity measure from a crp matrix
    [TODO : to optimize and re-implement it by pythonic numpy way]

    [1]. Chen, N., Li, W., & Xiao, H. (2017). Fusing similarity functions for cover song identification.
         Multimedia Tools and Applications.
    """
    Nx = crp.shape[0]
    Ny = crp.shape[1]
    cum_matrix = np.zeros(crp.shape)
    #for the moment doing it in the matlab way
    for i in range(3,Nx):
        for j in range(3,Ny):
            if int(crp[i,j]) == 1:
                cum_matrix[i][j] = max([cum_matrix[i-1][j-1],
                                            cum_matrix[i-2][j-1] + crp[i-1][j],
                                            cum_matrix[i-1][j-2] + crp[i][j-1],
                                            cum_matrix[i-3][j-1] + crp[i-2][j] + crp[i-1][j],
                                            cum_matrix[i-1][j-3] + crp[i][j-2], crp[i][j-1]]) + 1
            else:
                cum_matrix[i][j] = max([0,
                                            (cum_matrix[i-1][j-1] - gamma_state(crp[i-1][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-2][j-1] + crp[i-1][j]) - gamma_state(crp[i-2][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-1][j-2] + crp[i][j-1]) - gamma_state(crp[i-1][j-2], gammaO, gammaE)),
                                            ((cum_matrix[i-3][j-1] + crp[i-2][j] + crp[i-1][j]) -  gamma_state(crp[i-3][j-1], gammaO, gammaE)),
                                            ((cum_matrix[i-1][j-3] + crp[i][j-2] + crp[i][j-1]) -  gamma_state(crp[i-1][j-3], gammaO, gammaE))])
    #print "Cumulative Matrix computed with shape :", cum_matrix.shape
    dmax = np.divide(np.sqrt(Ny), np.max(cum_matrix))
    return dmax, cum_matrix
