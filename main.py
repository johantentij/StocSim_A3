import numpy as np
import matplotlib.pyplot as plt

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
