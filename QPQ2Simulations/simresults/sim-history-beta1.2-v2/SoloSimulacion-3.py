#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# VERSION 2020/02/19 11:45

import math
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta
from scipy.ndimage.interpolation import shift
import seaborn as sns

# Values is a matrix (N x r) or (number of playes x number of rounds)
def QPQ(declaredVals, trueVals, histlen, maxpval, KSTest=True, debug=True):
    dims = declaredVals.shape
    N = dims[0] # number of players
    R = dims[1] # number of rounds
    decisions = np.full(R, np.nan)
    historic = np.full((N, histlen), np.nan) # Empty history matrix (N x histlen) or (number of playes x History Len)
    
    utilities = np.zeros((N, R)) # Empty utility matrix (N x R)
    falsenegatives = np.zeros((N, R)) # Empty utility matrix (N x R)
    for i in range(R):
        # Roll historic to the left
        if (i > histlen):
            historic = np.roll(historic, -1, axis=1)
            
        theta = np.zeros(N)
        # copy declared values at the end of the historic
        historic[:, min(i, histlen - 1)] = declaredVals[: , i]
        
        for j in range(N):
            if debug: 
                print ("player ", j, " has values ", historic[j])
            if KSTest and stats.kstest(historic[j, 0:min(i+1, histlen)], 'uniform').pvalue < (1 - maxpval):
                if debug: print("False negative")
                theta[j] = np.random.uniform(0, 1, 1)
                falsenegatives[j, i] = 1
            else:
                theta[j] = declaredVals[j, i]
                
        # copy declared values at the end of the historic
        historic[:, min(i, histlen - 1)] = theta
        
        d = int(np.argmin(theta))
        if debug: print("Win player ", d)
        decisions[i] = d
        utilities[d, i] = trueVals[d, i]
    return decisions, utilities, falsenegatives



# Values is a matrix (N x r) or (number of playes x number of rounds)
# Level 0 is base level 
# Level 1 is cluster level
def QPQ2Level(declaredVals, trueVals, K, histlenL0, maxpvalL0, histlenL1, maxpvalL1, KSTest=True, debug=True):
    dims = declaredVals.shape
    N = dims[0] # number of players
    R = dims[1] # number of rounds
    playersPerCluster = int(N/K)
    if debug:
        print("Number of players ", N)
        print("Number of clusters ", K)
        print("Number of players per Cluster ", playersPerCluster)
    decisions = np.full(R, np.nan)
    historicL0 = np.full((N, histlenL0), np.nan) # Empty history matrix (N x histlen) or (number of playes x History Len)
    historicL1 = np.full((K, histlenL1), np.nan)
    utilities = np.zeros((N, R)) # Empty utility matrix (N x R)
    falsenegativesL0 = np.zeros((N, R)) # Empty utility matrix (N x R)
    falsenegativesL1 = np.zeros((K, R)) # Empty utility matrix (K x R)
    falsenegativesBoth = np.zeros((N, R)) # Empty utility matrix (N x R)
    for i in range(R):
        # Roll historic to the left
        if (i > histlenL0):
            historicL0 = np.roll(historicL0, -1, axis=1)
            
        theta = np.zeros(N)
        # copy declared values at the end of the historic
        historicL0[:, min(i, histlenL0 - 1)] = declaredVals[: , i]

        for j in range(N):
            if debug: 
                print ("player ", j, " has values ", historicL0[j])
            if KSTest and stats.kstest(historicL0[j, 0:min(i+1, histlenL0)], 'uniform').pvalue < (1 - maxpvalL0):
                if debug: print("False negative")
                theta[j] = np.random.uniform(0, 1, 1)
                falsenegativesL0[j, i] = 1
            else:
                theta[j] = declaredVals[j, i]
                
        # copy declared values at the end of the historic
        historicL0[:, min(i, histlenL0 - 1)] = theta
        
        thetaUp = np.zeros(K)
        decisionsL1 = np.full(K, np.nan)
        for k in range(K):     
            valsCluster = theta[(k)*playersPerCluster:(k+1)*playersPerCluster]
            decisionsL1[k] = int(np.argmin(valsCluster))
            player = int((k)*playersPerCluster + decisionsL1[k])
            if debug: 
                print ("cluster ", k, " has values ", valsCluster)
                print ("cluster ", k, " has player ", player)
                print ("cluster ", k, " has value ", valsCluster[int(decisionsL1[k])])
                print ("cluster ", k, " has range ", (k, min(i, histlenL1 - 1)))
            historicL1[k, min(i, histlenL1 - 1)] = valsCluster[int(decisionsL1[k])]
            if KSTest and stats.kstest(historicL1[k, 0:min(i+1, histlenL1)], stats.beta(a=1., b=playersPerCluster).cdf).pvalue < (1 - maxpvalL1):
                if debug: print("False negative")
                thetaUp[k] = np.random.uniform(0, 1, 1)
                falsenegativesL1[k, i] = 1
                falsenegativesBoth[player, i] = 1
            else:
                thetaUp[k] = valsCluster[int(decisionsL1[k])]
                
        #print(thetaUp)        
        dCluster = np.argmin(thetaUp)
        d = int(dCluster*playersPerCluster + decisionsL1[dCluster])
        if debug: print("Win cluster ", dCluster, " and player ", d)
        decisions[i] = d
        utilities[d, i] = trueVals[d, i]
    return decisions, utilities, falsenegativesL0, falsenegativesL1, falsenegativesBoth



def thresholdLevel(x, a, th0=0.97, x0=100): 
    #print("thresholdLevel a = ", a)
    return (th0)**((x0/x)**(1/math.log2(a)))

def doSimulation(numplayers, numLiars, numclusters, betaf, thresholdFunct, historyLen):
    # One entry for each simulation
    simresults = {'players': [], 'clusters': [], 'playerperclusters': [], 'numliars': [],
                  'QPQversion': [], 'alpha': [], 'rounds': [], 'betafactor': [],
                  'QPQHL': [], 'QPQTH': [], 'QPQ2HL0': [], 'QPQ2TH0': [], 'QPQ2HL1': [], 'QPQ2TH1': [],
                  'UtilityHonest': [], 'UtilityDishonest': [], 'FNTotalHonest': [], 'FNTotalDishonest': [],
                  'UtilityHonest_HL': [], 'UtilityDishonest_HL': [], 'FNTotalHonest_HL': [], 'FNTotalDishonest_HL': [] 
                }

    # historyLen = historyLenArray[0]
    if (numclusters <= 0):
        numclusters = 1
    print("Simulation using numplayers =", numplayers, " numclusters = ", numclusters, " numLiars = ", numLiars, " betafactor = ", betaf, " and historyLen = ", historyLen)
    playersPerCluster = int(numplayers/numclusters)
    threshold = thresholdFunct(historyLen)

    hlL0 = int((historyLen * numplayers) / (numclusters + alpha * numplayers / numclusters))
    hlL1 = int(alpha * hlL0)

    thresholdL0 = thresholdFunct(hlL0)
    thresholdL1 = thresholdFunct(hlL1)
    if (debug):
        print("    Using historyLen at QPQ = ", historyLen)
        print("    Using historyLen L0 at QPQ2 =", hlL0, " and historyLen L1 at QPQ2 =", hlL1)
        print("    Using threshold at QPQ =", threshold)
        print("    Using threshold L0 at QPQ2 =", thresholdL0, " and threshold L1 at QPQ2 =", thresholdL1)
    for k in range(numberSimulations):
        if (debug and (k % 100 == 0)):
            print("Simulation number =", k)
        trueVals = np.random.uniform(0, 1, (numplayers, rounds))
        declaredVals = np.array(trueVals, copy=True)
        liarsPos = np.full(numplayers, False)

        if (numLiars > 0):
            liarsPos[:numLiars] = True
            np.random.shuffle(liarsPos)
            norm = stats.distributions.beta(betaf, 1)
            declaredVals[liarsPos, :] = norm.ppf(trueVals[liarsPos, :])
         
        for version in [1,2]:
            # Scenario parameters
            simresults['players'].append(numplayers)
            simresults['clusters'].append(numclusters)
            simresults['numliars'].append(numLiars)
            simresults['QPQversion'].append(version)
            if (version == 1):
                simresults['alpha'].append(float('NaN'))
            else:
                simresults['alpha'].append(alpha)
            simresults['rounds'].append(rounds)
            simresults['betafactor'].append(betaf)
            simresults['QPQHL'].append(historyLen)
            
            # Scenario variables (computed)
            simresults['QPQTH'].append(threshold)
            if (version == 1):
                simresults['playerperclusters'].append(numplayers)
                simresults['QPQ2HL0'].append(float('NaN'))
                simresults['QPQ2HL1'].append(float('NaN'))
                simresults['QPQ2TH0'].append(float('NaN'))
                simresults['QPQ2TH1'].append(float('NaN'))
            else:
                simresults['playerperclusters'].append(playersPerCluster)
                simresults['QPQ2HL0'].append(hlL0)
                simresults['QPQ2HL1'].append(hlL1)
                simresults['QPQ2TH0'].append(thresholdL0)
                simresults['QPQ2TH1'].append(thresholdL1)
            
            if (version == 1):    
                rstT, utT, fnT = QPQ(declaredVals, trueVals, historyLen, threshold, KSTest=True, debug=False)
            else:
                rstT, utT, fnpL0T, fnL1T, fnT  = QPQ2Level(declaredVals, trueVals, numclusters, hlL0, thresholdL0, hlL1, thresholdL1, KSTest=True, debug=False)
                
            # Scenario results (simulation results) Note that we store the mean of players utility or FN rate
            simresults['UtilityHonest'].append(utT[~liarsPos].mean())
            simresults['FNTotalHonest'].append(fnT[~liarsPos].mean())
            if (numLiars > 0):
                simresults['UtilityDishonest'].append(utT[liarsPos].mean())
                simresults['FNTotalDishonest'].append(fnT[liarsPos].mean())
            else:
                simresults['UtilityDishonest'].append(float('NaN'))
                simresults['FNTotalDishonest'].append(float('NaN'))
                
            simresults['UtilityHonest_HL'].append(utT[numLiars:, historyLen:].mean())
            simresults['FNTotalHonest_HL'].append(fnT[numLiars:, historyLen:].mean())
            if (numLiars > 0):
                simresults['UtilityDishonest_HL'].append(utT[0:numLiars, historyLen:].mean())
                simresults['FNTotalDishonest_HL'].append(fnT[0:numLiars, historyLen:].mean())
            else:
                simresults['UtilityDishonest_HL'].append(float('NaN'))
                simresults['FNTotalDishonest_HL'].append(float('NaN'))
     
    #print(simresults)   
    return pd.DataFrame(data=simresults)


debug = True
dir = './results'

# Number of simulations
numberSimulations = 50

# History length
historyLenArray = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] 
alpha = 1 


# Number of players
numplayersArray = [64]

# betafactor (dishonest beta factor) Array
betafactorArray = [1.2]

# Number of Liars
numLiarsArray = [1]

# Number of clusters
numclustersArray = [8]


if not os.path.exists(dir):
    os.mkdir(dir)
frames=[]    

for count in range(100): # in total  100*50 = 5.000 simulations for each scenario
    for idx0, historyLen in enumerate(historyLenArray):
        for idx1, numclusters in enumerate(numclustersArray):
            for idx2, numplayers in enumerate(numplayersArray):
                for idx3, betaFactor in enumerate(betafactorArray):
                    for idx4, numLiars in enumerate(numLiarsArray):
                        if (int(numplayers/numclusters) < 2 or numplayers <= numLiars):
                            continue

                        # Number of rounds
                        rounds = historyLen*10

                        a = numplayers * numclusters / ( numclusters ^ 2 + numplayers )
                        tfnew = lambda hl: thresholdLevel(hl, a)

                        simRst = doSimulation(numplayers, numLiars, numclusters, betaFactor, tfnew, historyLen)
                        frames.append(simRst)

                        timestr = time.strftime("%Y%m%d-%H%M%S")
                        file_name = 'QPQRst-{0}-{1}-{2}-{3}-{4}-{5}-{6}.csv'.format(str(numplayers),str(numclusters),str(rounds),str(betaFactor),str(numLiars),str(historyLen), timestr)
                        full_path = os.path.join(dir, file_name)
                        simRst.to_csv(full_path, index=False)

df = pd.concat(frames, ignore_index=True)
full_path = os.path.join(dir, 'QPQRst-all-{0}.csv'.format(timestr))
df.to_csv(full_path, index=False)
