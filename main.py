import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from grid import GridGen

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


def uniformDie(u):
    return int(6 * u) + 1

def geometricDie(u, p=0.5):
    return int(np.floor(np.log(1 - u) / np.log(1 - p))) + 1

def levyDie(u):
    if u < 0.9:
        v = u / 0.9
        return int(6 * v) + 1
    else:
        v = (u - 0.9) / 0.1   
        return int(21 * v) + 10


def playGame(die, jumps):
    pos = 0
    turnCount = 0
    jumpCount = 0
    while (pos != 100):
        u = np.random.random()
        nextPos = pos + die(u)

        jumpPos = (nextPos == jumps[:, 0])
        if np.sum(jumpPos):
            nextPos = jumps[jumpPos, 1]
            jumpCount += 1

        if (nextPos <= 100):
            pos = nextPos

        turnCount += 1

    return turnCount, jumpCount


def playAntitheticGame(die, jumps, board_size=100):
    pos_1, pos_2 = 0, 0
    turnCount_1, turnCount_2 = 0, 0

    def check_jump(u, pos):
        nextPos = pos + die(u)

        if nextPos > board_size:
            return pos

        jumpPos = (nextPos == jumps[:, 0])
        if np.any(jumpPos):
            nextPos = jumps[jumpPos, 1][0]

        return nextPos

    while pos_1 != board_size or pos_2 != board_size:
        u = np.random.random()

        if pos_1 != board_size:
            pos_1 = check_jump(u, pos_1)
            turnCount_1 += 1

        if pos_2 != board_size:
            pos_2 = check_jump(1 - u, pos_2)
            turnCount_2 += 1

    return turnCount_1, turnCount_2


# Histogram of game length for random grids
N = 10000
p = 0.2
die = lambda u: uniformDie(u)
turnDist = np.empty(N)
jumpDist = np.empty(N)
gg = GridGen(ladders, snakes,100)
for i in range(N):
    new_ladders, new_snakes = gg.gen_grid()
    new_jumps = np.concatenate((new_ladders, new_snakes))
    turnDist[i], jumpDist[i] = playGame(die, new_jumps)

plt.hist(turnDist, bins=100, density=True)
plt.title("Number of turns before game ends")
plt.xlabel("Number of turns")
plt.ylabel("Density of occurence")
plt.show()


# Game length normal vs levy die
N = 10000
turns_normal = np.zeros(N)
times_normal = np.zeros(N)
turns_levy = np.zeros(N)
times_levy = np.zeros(N)

for i in range(N):
    t, time = playGame(uniformDie, jumps)
    turns_normal[i] = t
    times_normal[i] = time
    
    t, time = playGame(levyDie, jumps)
    turns_levy[i] = t
    times_levy[i] = time

mean_normal = np.mean(turns_normal)
mean_levy = np.mean(turns_levy)
print(mean_levy)
print(mean_normal)


# Antithetic covariance test
N = 1000000
T1 = np.zeros(N)
T2 = np.zeros(N)

for i in range(N):
    t1, t2 = playAntitheticGame(levyDie, jumps)
    T1[i] = t1
    T2[i] = t2

cov = np.cov(T1, T2, ddof=1)[0, 1]
corr = np.corrcoef(T1, T2)[0, 1]

print("Covariance:", cov)
print("Correlation:", corr)


# Stat tests
sem_normal = stats.sem(turns_normal)
sem_levy = stats.sem(turns_levy)

ci_normal = stats.t.interval(0.95, len(turns_normal)-1, loc=mean_normal, scale=sem_normal)
ci_levy = stats.t.interval(0.95, len(turns_levy)-1, loc=mean_levy, scale=sem_levy)

t_stat, p = stats.ttest_ind(turns_normal, turns_levy, equal_var=False)

if p < 0.05:
    print("Significant statistical difference found.")
else:
    print("No significant statistical difference found.")

