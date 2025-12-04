import numpy as np
import matplotlib.pyplot as plt
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

def uniformDie():
    return np.random.randint(1, 7)

def geometricDie(p=0.5):
    return np.random.geometric(p)

def levyDie():
    p = np.random.random()
    
    if p < 0.90:
        return np.random.randint(1, 7)
    
    else:
        return np.random.randint(10, 31) 

def playGame(die):
    pos = 0

    turnCount = 0
    jumpCount = 0
    while (pos != 100):
        nextPos = pos + die()

        jumpPos = (nextPos == jumps[:, 0])
        if np.sum(jumpPos):
            nextPos = jumps[jumpPos, 1]
            jumpCount += 1

        if (nextPos <= 100):
            pos = nextPos

        turnCount += 1

    return turnCount, jumpCount

N = 1000
p = 0.2
die = lambda : geometricDie(p)

turnDist = np.empty(N)
jumpDist = np.empty(N)
for i in range(N):
    turnDist[i], jumpDist[i] = playGame(die)

plt.hist(turnDist, density=True)
plt.title("Number of turns before game ends")
plt.xlabel("Number of turns")
plt.ylabel("Density of occurence")
plt.show()



N = 1000

turns_normal = np.zeros(N)
times_normal = np.zeros(N)
turns_levy = np.zeros(N)
times_levy = np.zeros(N)

for i in range(N):
    t, time = playGame(uniformDie)
    turns_normal[i] = t
    times_normal[i] = time
    
    t, time = playGame(levyDie)
    turns_levy[i] = t
    times_levy[i] = time

mean_normal = np.mean(turns_normal)
mean_levy = np.mean(turns_levy)
print(mean_levy)
print(mean_normal)

sem_normal = stats.sem(turns_normal)
sem_levy = stats.sem(turns_levy)

ci_normal = stats.t.interval(0.95, len(turns_normal)-1, loc=mean_normal, scale=sem_normal)
ci_levy = stats.t.interval(0.95, len(turns_levy)-1, loc=mean_levy, scale=sem_levy)

t_stat, p = stats.ttest_ind(turns_normal, turns_levy, equal_var=False)

if p < 0.05:
    print("Significant statistical difference found.")
else:
    print("No significant statistical difference found.")
