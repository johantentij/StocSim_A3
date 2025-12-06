import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import scipy.sparse as sp

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

   
    for j in range(100):
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

def exactExpectedLength(p):
    T = sp.csr_array((107, 101))
    I = sp.diags(np.ones(100), 0)

    for i in range(6):
        T += p[i] * T_terms[i]

    T = makeSquare.dot(T) + sp.diags(overshoot.dot(T).toarray(), 0)
    T.resize((100, 100))

    E = sp.linalg.spsolve(I - T.T, np.ones(100))

    return E[0]

def step(prevL, p, perturbation, temperature):
    # propose a new die
    proposed = p + np.random.normal(0, perturbation, 6)
    proposed[proposed < 0] = 0
    proposed /= np.sum(proposed)

    newL = exactExpectedLength(proposed)

    alpha = np.min((1, np.exp(-(newL - prevL) / temperature)))

    if np.random.rand() <= alpha:
        return newL, proposed
    else:
        return prevL, p
    
def MetropolisHastings(p_0, perturbation, steps=1000, temperature=10):
    p = p_0 / np.sum(p_0)

    gameLengths = np.empty(steps)
    gameLengths[0] = exactExpectedLength(p)
    for i in range(1, steps):
        gameLengths[i], p = step(gameLengths[i - 1], p, perturbation, temperature)

    return gameLengths, p

gameLengths, p = MetropolisHastings(np.ones(6), 1e-2, temperature=0.1)

plt.plot(gameLengths[1:])
plt.title("Metropolis-Hastings")
plt.xlabel("Steps")
plt.ylabel("Expected game length")
plt.figure()

plt.bar(np.arange(6) + 1, p)
plt.title("PMF of optimised die")
plt.xlabel("Roll")
plt.ylabel("Probability")
plt.show()



