# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

"""
Functions' declaration.
"""


def gaussian(var, mean, sd):
    return np.exp(-0.5 * np.power(var - mean, 2.0) / (sd * sd)) / (sd * np.sqrt(2 * np.pi))


def integrand(var, mean, sd, cr):
    return 0.016 * np.exp(0.5 * 0.204 * 0.204) * np.power(2 * var * cr, 2.013) * gaussian(var, mean, sd)


def above_ground_biomass(mean, sd, cr):
    return integrate.quad(lambda v: integrand(v, mean, sd, cr), 0, np.inf)


"""
Simulation
"""

rng = np.random.default_rng()

minHMean = 5.0
maxHMean = 45.0
minHSd = 5.0
maxHSd = 18.0

gridSize = 10
meanVec = np.linspace(minHMean, maxHMean, gridSize)
sdVec = np.linspace(minHSd, maxHSd, gridSize)
nSamples = 100

errorMatrix = np.zeros((0, 4))

pctCorrect = [70.0]
nBins = [10]

for n in nBins:
    meanDiscret = np.linspace(minHMean, maxHMean, n)
    sdDiscret = np.linspace(minHSd, maxHSd, n)
    for pc in pctCorrect:
        p = pc * 0.01
        for m in meanVec:
            correctBinMean = np.max(np.where(meanDiscret <= m)[0])
            for s in sdVec:
                correctBinSd = np.max(np.where(sdDiscret <= s)[0])

                estimatedAGB = np.zeros(nSamples)
                correctAGB, err = above_ground_biomass(m, s, 1.0)
                if correctAGB == 0:
                    print(str(m) + ", " + str(s))

                for ns in range(nSamples):
                    # print("Mean: " + str(m) + ", Std: " + str(s) + ", samples:  " str(ns) + " out of " + str(
                    # nSamples))
                    x = rng.random()
                    if x < p:
                        hMean = meanDiscret[correctBinMean]
                    elif x < p + (1 - p) * 0.5:
                        if correctBinMean == 0:
                            hMean = meanDiscret[correctBinMean + 1]
                        else:
                            hMean = meanDiscret[correctBinMean - 1]
                    else:
                        if correctBinMean == np.size(meanDiscret) - 1:
                            hMean = meanDiscret[correctBinMean - 1]
                        else:
                            hMean = meanDiscret[correctBinMean + 1]
                    x = rng.random()
                    if x < p:
                        hSd = sdDiscret[correctBinSd]
                    elif x < p + (1 - p) * 0.5:
                        if correctBinSd == 0:
                            hSd = sdDiscret[correctBinSd + 1]
                        else:
                            hSd = sdDiscret[correctBinSd - 1]
                    else:
                        if correctBinSd == np.size(sdDiscret) - 1:
                            hSd = sdDiscret[correctBinSd - 1]
                        else:
                            hSd = sdDiscret[correctBinSd + 1]

                    agb, errEst = above_ground_biomass(hMean, hSd, 1.0)
                    estimatedAGB[ns] = agb

                errorAGB = (correctAGB - estimatedAGB) / correctAGB * 100.0
                absErrorAGB = np.abs(correctAGB - estimatedAGB) / correctAGB * 100.0

                avgError = np.mean(errorAGB)
                avgAbsError = np.mean(absErrorAGB)

                new_row = np.array([[m, s, avgError, avgAbsError]])
                errorMatrix = np.append(errorMatrix, new_row, axis=0)

errorMatrix = np.nan_to_num(errorMatrix, nan=np.nan, posinf=np.nan, neginf=np.nan)

# build errors for plotting
axMean, axSd = np.meshgrid(meanVec, sdVec)

zAvgError = np.zeros((gridSize, gridSize))
zAvgAbsError = np.zeros((gridSize, gridSize))

for i, x in enumerate(meanVec):
    ix = np.where(errorMatrix[:, 0] == x)[0]
    for j, y in enumerate(sdVec):
        jx = np.where(errorMatrix[:, 1] == y)[0]
        # print(ix)
        # print(jx)
        kx = ix[np.isin(ix, jx)][0]
        # print(kx)
        avgError = errorMatrix[kx, 2]
        avgAbsError = errorMatrix[kx, 3]

        zAvgError[i, j] = avgError
        zAvgAbsError[i, j] = avgAbsError

print(errorMatrix)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
CS1 = axs[0].contourf(axMean, axSd, zAvgError, 10, origin="lower")
CS2 = axs[1].contourf(axMean, axSd, zAvgAbsError, 10, origin="lower")
fig.colorbar(CS1)
fig.colorbar(CS2)
plt.show()

# print(errorMatrix)
