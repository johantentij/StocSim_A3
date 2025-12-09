import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import scipy.sparse as sp

# snakes = np.array([[17, 7],
#                    [54, 34],
#                    [62, 19],
#                    [64, 60],
#                    [93, 73],
#                    [95, 75],
#                    [98, 79]])

# ladders = np.array([[1, 38], 
#                     [4, 14], 
#                     [9, 31],
#                     [21, 42],
#                     [28, 84],
#                     [51, 67],
#                     [71, 91],
#                     [80, 99]])

# more cheating resistant board
snakes = np.array([[27,  7],
                   [50, 30],
                   [55, 35],
                   [68, 48],
                   [71, 61],
                   [77, 67],
                   [94, 84]])

ladders = np.array([[4, 25],
                    [10, 32],
                    [36, 52],
                    [43, 80],
                    [46, 66],
                    [63, 73],
                    [64, 83],
                    [75, 85]])

# board with best die = always roll 1
# snakes = np.array([[36, 32],
#  [54, 34],
#  [61, 51],
#  [66, 62],
#  [67, 48],
#  [93, 74],
#  [95, 75]])

# ladders = np.array([[3, 13],
#  [17, 73],
#  [24, 43],
#  [31, 41],
#  [49, 68],
#  [57, 78],
#  [81, 97],
#  [82, 92]])


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

def exactExpectedLength(p):
    T = sp.csr_array((107, 101))
    I = sp.diags(np.ones(100), 0)

    for i in range(6):
        T += p[i] * T_terms[i]

    T = makeSquare.dot(T) + sp.diags(overshoot.dot(T).toarray(), 0)
    T.resize((100, 100))

    try:
        E = sp.linalg.spsolve(I - T.T, np.ones(100))
        result = E[0]
        
        if result < 0 or result > 1e6:
            return 1e9
            
        return result
        
    except RuntimeError:
        return 1e9 
    except Exception:
        return 1e9

def step(prevAdvantage, basePmf, p, perturbation, temperature):
    # propose a new die
    proposed = p + np.random.normal(0, perturbation, 6)
    proposed[proposed < 0] = 0
    proposed /= np.sum(proposed)

    newPmf = makeDistribution(proposed)
    newAdvantage = winningPercentage(newPmf, basePmf)

    alpha = np.min((1, np.exp((newAdvantage - prevAdvantage) / temperature)))

    if np.random.rand() <= alpha:
        return newAdvantage, proposed
    else:
        return prevAdvantage, p
    
def MetropolisHastings(p_0, perturbation=1e-2, steps=1000, gradientFunc=None, temp=1e-3):
    if (gradientFunc is None):
        # if no gradient is given use constant temperature
        gradientFunc = lambda x : temp

    p = p_0 / np.sum(p_0)

    basePmf = makeDistribution(np.ones(6) / 6)
    advantages = np.empty(steps)
    advantages[0] = winningPercentage(makeDistribution(p), basePmf)

    for i in range(1, steps):
        # propose a new die
        proposed = p + np.random.normal(0, perturbation, 6)
        proposed[proposed < 0] = 0
        proposed /= np.sum(proposed)

        newPmf = makeDistribution(proposed)
        newAdvantage = winningPercentage(newPmf, basePmf)

        temp = gradientFunc(i)
        alpha = np.min((1, np.exp((newAdvantage - advantages[i - 1]) / temp)))

        if np.random.rand() <= alpha:
            p = proposed
            advantages[i] = newAdvantage
        else:
            advantages[i] = advantages[i - 1]

    return advantages, p

def expGradient(i, steps, start, end):
    return start * np.power(end / start, i / (steps - 1))

def linGradient(i, steps, start, end):
    return start + (end - start) * i / (steps - 1)


N = 50
steps = 1000
advantages = np.empty((N, steps))
dies = np.empty((N, 6))

gradientFunc = lambda i : expGradient(i, steps, 1e-1, 1e-4)

for i in range(N):
    print("run %d out of %d" % (i, N))
    p_0 = np.random.random(6)
    p_0 /= np.sum(p_0)

    advantages[i], dies[i] = MetropolisHastings(p_0, steps=steps, gradientFunc=gradientFunc)

# simulated annealing plot
for i in range(N):
    plt.plot(advantages[i, :], color="C0", alpha=0.2)

plt.title("Metropolis-Hastings")
plt.xlabel("Steps")
plt.ylabel("Winning chance with biased die")
plt.figure()

best = np.argmax(advantages[:, -1])
bestDie = dies[best, :]
meanThrow = np.sum(np.arange(1, 7) * bestDie)
print("best advantage: %f" % advantages[best, -1])

plt.bar(np.arange(6) + 1, bestDie)
plt.vlines(meanThrow, 0, np.max(bestDie), linestyle="dashed", color="grey")
plt.title("PMF of optimised die")
plt.xlabel("Roll")
plt.ylabel("Probability")
plt.show()

# plot game length PMF distributions
# distRef = makeDistribution(np.ones(6) / 6)
# dist = makeDistribution(p)

# print(winningPercentage(p, np.ones(6) / 6))

# plt.plot(distRef, label="fair die")
# plt.plot(dist, label="optimised die")
# plt.xlabel("Number of turns")
# plt.ylabel("Probability of game ending")
# plt.legend()
# plt.show()

# plot game length PMF distributions
# distRef = makeDistribution(np.ones(6) / 6)
# dist = makeDistribution(p)

# print(winningPercentage(p, np.ones(6) / 6))

# plt.plot(distRef, label="fair die")
# plt.plot(dist, label="optimised die")
# plt.xlabel("Number of turns")
# plt.ylabel("Probability of game ending")
# plt.legend()
# plt.show()
