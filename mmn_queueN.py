#!/usr/bin/env python3

import argparse
import collections
import logging
import matplotlib.pyplot as plt
from random import expovariate, seed
import random
from discrete_event_sim import Simulation, Event


class MMN(Simulation):

    def __init__(self, lambd, mu, n, d, max_t):
        if n < 1 or d > n:
            raise NotImplementedError

        super().__init__()
        self.d = d
        self.queues = [collections.deque() for _ in range(n)]
        self.times = []
        self.running = [None] * n
        self.arrivals = {}
        self.completions = {}
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.max_t = max_t
        self.arrival_rate = lambd * n
        self.array_len = [0] * self.n
        self.schedule(expovariate(self.arrival_rate), Arrival(0, 0))
        self.interval = 10
        self.schedule(0, SavingTime())

    def schedule_arrival(self, job_id):
        index = self.super_market()
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id, index))

    def schedule_completion(self, job_id, index):
        self.schedule(expovariate(self.mu), Completion(job_id, index))

    def super_market(self):
        sample = random.sample(self.queues, self.d)
        shortest_lists = [lst for lst in sample if len(lst) == min(len(sublist) for sublist in sample)]
        selected_list = random.choice(shortest_lists)
        index = self.queues.index(selected_list)

        return index

    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queues[i])


class SavingTime(Event):

    def process(self, sim: MMN):

        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))

        sim.schedule(sim.interval, SavingTime())


class Arrival(Event):

    def __init__(self, job_id, index):
        self.id = job_id
        self.index = index

    def process(self, sim: MMN):  # TODO: complete this method
        sim.arrivals[self.id] = sim.t
        # if sim.running[self.id] is None:
        if sim.running[self.index] is None:
            sim.running[self.index] = self.id
            sim.schedule_completion(self.id, self.index)
        else:
            sim.queues[self.index].append(self.id)

        sim.schedule_arrival(self.id + 1)


class Completion(Event):
    def __init__(self, job_id, index):
        self.id = job_id
        self.index = index

    def process(self, sim: MMN):  # TODO: complete this method
        assert sim.running[self.index] is not None
        sim.completions[self.id] = sim.t

        # if self.id in sim.running:
        sim.running[self.index] = None

        if sim.queues[self.index]:
            popped_job = sim.queues[self.index].popleft()
            sim.running[self.index] = popped_job
            sim.schedule_completion(popped_job, self.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5, 0.9, 0.95, 0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout
    styles = ['-', '--', '-.', ':']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(10, 6))

    for i, lambd_value in enumerate(args.lambd):
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t)
        sim.run(args.max_t)

        completions = sim.completions

        print(len(sim.arrivals))

        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")
        print(f"Theoretical expectation for random server choice: {1 / (1 - lambd_value)}")

        counts = [0] * 15
        queueLengths = sim.times

        for length in queueLengths:
            if length == 0:
                continue
            for t in range(min(length, 15)):
                counts[t] += 1
        fractions = [count / len(queueLengths) for count in counts]

        plt.plot(range(1, 15), fractions[1:], color=colors[i % len(colors)], label=f'Theoretical λ = {lambd_value}',
                 linestyle='dotted')

    plt.xlabel('Queues Length')
    plt.ylabel('Fraction of Queues with at least that size')
    plt.title(f"Theoretical Queues Length | n: {args.n} | d: {args.d} ")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, 16)
    plt.xticks(range(0, 16))
    plt.show()


if __name__ == '__main__':
    main()
