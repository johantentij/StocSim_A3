import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy import stats

snakes = np.array([[17, 7],
                   [54, 34],
                   [62, 19],
                   [64, 60],
                   [93, 73],
                   [95, 75],
                   [98, 79]])

ladders = np.array([[1, 38], 
                    [4, 14], 
                    [9, 31],
                    [21, 42],
                    [28, 84],
                    [51, 67],
                    [71, 91],
                    [80, 99]])

jumps = np.concatenate((snakes, ladders))

T_terms = []
for k in range(6):
    T = np.zeros((107, 101))

   
    for j in range(101):
        jumpPos = (j + k + 1 == jumps[:, 0])
        if np.sum(jumpPos):
            T[jumps[jumpPos, 1], j] = 1
        else:
            T[j + k + 1 , j] = 1

    T_terms.append(sp.csr_array(T))

overshoot = np.zeros(107)
overshoot[-6:] = 1
overshoot = sp.csr_array(overshoot)

makeSquare = np.concatenate((np.identity(101), np.zeros((101, 6))), axis=1)
makeSquare = sp.csr_array(makeSquare)

def makeDistribution(p, maxThrows=500):
    T = sp.csr_array((107, 101))

    for i in range(6):
        T += p[i] * T_terms[i]

    overshootVector = overshoot.dot(T).toarray()
    overshootVector[-1] = 0
    T = makeSquare.dot(T) + sp.diags(overshootVector, 0)

    v = np.zeros(101)
    v[0] = 1

    dist = np.zeros(maxThrows)
    i = 0
    for i in range(maxThrows):
        v = T.dot(v)
        dist[i] = v[-1]

        i += 1

    dist[i:] = 0

    return dist

def winningPercentage(pmfWinner, pmfLoser):
    cdfWinner = np.cumsum(pmfWinner)

    prob = 0
    for i in range(1, np.shape(pmfLoser)[0]):
        prob += pmfLoser[i] * (cdfWinner[i - 1] + .5 * pmfWinner[i])

    return prob

def die(p):
    return np.random.choice(np.arange(1, 7), p=p)

def playGame(p):
    pos = 0
    turnCount = 0
    while (pos != 100):
        nextPos = pos + die(p)

        jumpPos = (nextPos == jumps[:, 0])
        if np.sum(jumpPos):
            nextPos = jumps[jumpPos, 1]

        if (nextPos <= 100):
            pos = nextPos

        turnCount += 1

    return turnCount

# takes a while, use N = 100 for testing
N = 1000
Ngames = 1000

p_0 = np.ones(6) / 6
p_biased = np.array([0.1, 0, 0.3, 0, 0, 0.6])

def empiricalWinningChance(p_0, p_biased, N):
    wins = 0
    for i in range(Ngames):
        turnCountFair = playGame(p_0)
        turnCountBiased = playGame(p_biased)

        if (turnCountBiased < turnCountFair):
            wins += 1
        elif (turnCountBiased == turnCountFair):
            if (np.random.rand() > .5):
                wins += 1

    return wins / N

winningChances = np.empty(N)
for i in range(N):
    print("run %d out of %d" % (i, N))

    winningChances[i] = empiricalWinningChance(p_0, p_biased, Ngames)

fairPMF = makeDistribution(p_0)
biasedPMF = makeDistribution(p_biased)

advantage = winningPercentage(biasedPMF, fairPMF)

t_stat, p = stats.ttest_1samp(winningChances, advantage)

print("p-value:", p)

if p < 0.05:
    print("Significant statistical difference found.")
else:
    print("No significant statistical difference found.")

plt.hist(winningChances, density=True)
plt.xlabel("Winning chance")
plt.ylabel("Density of occurence")
plt.show()
