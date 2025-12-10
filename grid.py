import numpy as np
import matplotlib.pyplot as plt
import random

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


class GridGen():
    def __init__(self, ladders, snakes, board_size=100):
        self.ladders = ladders
        self.snakes = snakes
        self.jumps = np.concatenate((snakes, ladders))
        self.Lists = [self.ladders, self.snakes, self.jumps]
        self.board_size = board_size

        # Absolute distances
        get_dists = lambda L: [abs(y - x) for (x, y) in L]
        # 0: ladders, 1: snakes, 2: both
        self.distances = [get_dists(L) for L in self.Lists]

        # Regions: [1–49], [50–(board_size-1)]
        self.region_bounds = [(1, 49), (50, self.board_size - 1)]

        ladder_starts = self.ladders[:, 0]
        snake_starts  = self.snakes[:, 0]

        self.ladder_region_counts = [
            np.sum(ladder_starts <= 49),
            np.sum(ladder_starts >= 50)
        ]

        self.snake_region_counts = [
            np.sum(snake_starts <= 49),
            np.sum(snake_starts >= 50)
        ]

        # region-specific length distributions
        self.ladder_lengths_by_region = []
        self.snake_lengths_by_region = []

        for (low, high) in self.region_bounds:
            mask_l = (ladder_starts >= low) & (ladder_starts <= high)
            region_ladders = self.ladders[mask_l]
            ladder_lengths = [abs(y - x) for (x, y) in region_ladders]

            mask_s = (snake_starts >= low) & (snake_starts <= high)
            region_snakes = self.snakes[mask_s]
            snake_lengths = [abs(x - y) for (x, y) in region_snakes]

            if len(ladder_lengths) == 0:
                ladder_lengths = self.distances[0]
            if len(snake_lengths) == 0:
                snake_lengths = self.distances[1]

            self.ladder_lengths_by_region.append(ladder_lengths)
            self.snake_lengths_by_region.append(snake_lengths)
    
    # L_idx:
    # 0 : ladders
    # 1 : snakes
    # 2 : both 
    def bootstrap_sample(self, List):
        return random.choice(List)

    def _generate_jumps(self, is_snake, region_counts, lengths, used, max_tries):
        new_list = []

        for r in range(len(self.region_bounds)):  # region 0, 1
            target = region_counts[r]
            tries = 0
            low, high = self.region_bounds[r]

            region_lengths = lengths[r]

            while sum(
                1 for (start, _) in new_list
                if low <= start <= high
            ) < target:

                if tries > max_tries:
                    raise ValueError("error")
                tries += 1

                # Sample length by bootstrap
                length = self.bootstrap_sample(region_lengths)

                # Sample start in this region
                s_low, s_high = low, high
                if is_snake and s_low == 1:  # snake cannot start on 1
                    s_low = 2
                start = random.randint(s_low, s_high)

                # Compute end
                end = start - length if is_snake else start + length

                # Check validity
                if is_snake:
                    if end < 1:
                        continue
                else:
                    if end >= self.board_size:
                        continue

                # Avoid overlapping starts/ends
                if start in used or end in used:
                    continue

                used.add(start)
                used.add(end)
                new_list.append([start, end])

        return np.array(sorted(new_list, key=lambda x: x[0]))

    def gen_grid(self, max_tries=10000):
        used = set()
        # Generate snakes
        new_snakes = self._generate_jumps(
            is_snake=True,
            region_counts=self.snake_region_counts,
            lengths=self.snake_lengths_by_region,
            used=used,
            max_tries=max_tries
        )

        # Generate ladders
        new_ladders = self._generate_jumps(
            is_snake=False,
            region_counts=self.ladder_region_counts,
            lengths=self.ladder_lengths_by_region,
            used=used,
            max_tries=max_tries
        )

        return new_ladders, new_snakes

    def hist_jumps(self):
        dists = [x - y for (x, y) in self.jumps]
        plt.hist(dists, edgecolor='black')
        plt.xlabel("Jump size")
        plt.ylabel("Frequency")
        plt.show()

gg = GridGen(ladders, snakes)
new_ladders, new_snakes = gg.gen_grid()
print("New ladders:\n", new_ladders)
print("New snakes:\n", new_snakes)
