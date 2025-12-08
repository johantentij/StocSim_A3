import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from main import playGame, uniformDie
from dieOptimisation import exactExpectedLength

def run_convergence_check(max_sims=10000):
    print(f"Starting convergence check with {max_sims} simulations...")

    # exact length
    p_uniform = np.ones(6) / 6
    theoretical_mean = exactExpectedLength(p_uniform)
    print(f"Theoretical Mean (Markov Chain): {theoretical_mean:.4f}")

    # monte carlo simulation
    turns = np.zeros(max_sims)
    
    for i in range(max_sims):
        t, _ = playGame(uniformDie)
        turns[i] = t
        
    ns = np.arange(1, max_sims + 1)
    
    # mean
    running_mean = np.cumsum(turns) / ns
    
    # sem
    running_sq_mean = np.cumsum(turns ** 2) / ns
    running_variance = running_sq_mean - (running_mean ** 2)
    running_std = np.sqrt(running_variance)
    running_sem = running_std / np.sqrt(ns)

    # 95% CI
    ci_upper = running_mean + 1.96 * running_sem
    ci_lower = running_mean - 1.96 * running_sem

    plt.figure(figsize=(10, 6))
    plt.axhline(y=theoretical_mean, color='r', linestyle='-', linewidth=2, label=f'Exact Theoretical Mean ({theoretical_mean:.2f})')
    plt.plot(ns, running_mean, color='b', linewidth=1, label='Simulated Running Mean')
    
    plt.fill_between(ns, ci_lower, ci_upper, color='b', alpha=0.2, label='95% Confidence Interval')

    plt.title("Convergence of Monte Carlo Simulation to Theoretical Mean\n(Law of Large Numbers)")
    plt.xlabel("Number of Simulations (N)")
    plt.ylabel("Average Game Length (Turns)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.xscale('log') 
    
    plt.show()

if __name__ == "__main__":
    run_convergence_check(max_sims=20000)