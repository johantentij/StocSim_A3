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

overshoot = np.zeros((1, 107))
overshoot[0, -6:] = 1
overshoot = sp.csr_array(overshoot)

makeSquare = np.concatenate((np.identity(101), np.zeros((101, 6))), axis=1)
makeSquare = sp.csr_array(makeSquare)

def exactExpectedLength(p):
    T = sp.csr_array((107, 101))
    I = sp.diags(np.ones(100), 0)

    for i in range(6):
        T += p[i] * T_terms[i]

    T_square = makeSquare.dot(T) + sp.diags(overshoot.dot(T).toarray().ravel(), 0)
    T = T_square[:100, :100]

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

def step(prevL, p, perturbation, temperature):
    # propose a new die
    proposed = p + np.random.normal(0, perturbation, 6)
    proposed[proposed < 0] = 0
    proposed /= np.sum(proposed)

    newL = exactExpectedLength(proposed)

    temp_safe = max(temperature, 1e-10)
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

def SimulatedAnnealing(p_0, perturbation, steps=5000, temperature=10.0, cooling_rate=0.995):
    p = p_0 / np.sum(p_0)
    
    gameLengths = np.empty(steps)
    gameLengths[0] = exactExpectedLength(p)
    
    best_p = p.copy()
    best_len = gameLengths[0]

    for i in range(1, steps):
        gameLengths[i], p = step(gameLengths[i - 1], p, perturbation, temperature)
        
        if gameLengths[i] < best_len:
            best_len = gameLengths[i]
            best_p = p.copy()
            
        temperature *= cooling_rate

    return gameLengths, best_p

p_init = np.ones(6)
history, best_die = SimulatedAnnealing(p_init, perturbation=0.05, steps=5000, temperature=2.0, cooling_rate=0.995)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title("Simulated Annealing Convergence")
plt.xlabel("Iteration")
plt.ylabel("Expected Game Length")

plt.subplot(1, 2, 2)
plt.bar(np.arange(6) + 1, best_die, color='orange', alpha=0.7)
plt.title("Optimised Die PMF")
plt.xlabel("Roll")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()

print("-" * 30)
print("Optimization Results")
print("-" * 30)
print(f"Initial Expected Length: {exactExpectedLength(p_init / np.sum(p_init)):.4f}")
print(f"Minimum Expected Length: {exactExpectedLength(best_die):.4f}")
print("Optimised Die PMF:")
print(np.round(best_die, 4))
print("-" * 30)
