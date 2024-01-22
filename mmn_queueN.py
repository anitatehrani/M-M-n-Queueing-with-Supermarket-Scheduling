#!/usr/bin/env python3

import argparse
import csv
import collections
from gettext import npgettext
import logging
from random import expovariate, seed
import matplotlib.pyplot as plt
from collections import Counter
from discrete_event_sim import Simulation, Event
import random
from math import factorial
from random import expovariate, seed
import collections
import random
from math import factorial
from discrete_event_sim import Simulation, Event


# One possible modification is to use a different distribution for job sizes or and/or interarrival times.
# Weibull distributions (https://en.wikipedia.org/wiki/Weibull_distribution) are a generalization of the
# exponential distribution, and can be used to see what happens when values are more uniform (shape > 1,
# approaching a "bell curve") or less (shape < 1, "heavy tailed" case when most of the work is concentrated
# on few jobs).

# To use weibull variates, for a given set of parameters do something like
# from workloads import weibull_generator
# gen = weibull_generator(shape, mean)

# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd, mu, n, d, max_t):
        if n < 1 or d > n:
            raise NotImplementedError  # extend this to make it work for multiple queues and supermarket

        super().__init__()
        self.d = d

        self.queue = []
        array = {}
        for i in range(n):
            array[i] = {'id': i}
            # array[i] = []
        self.queue.append(array)

        # print(self.queue)
        # exit()

        # self.running = [collections.deque() for i in range(n)]
        self.running = []
        # self.queue = 1  # FIFO queue of the system
        # self.queue = collections.deque()  # FIFO queue of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.max_t = max_t
        self.arrival_rate = lambd * n
        self.schedule(expovariate(self.arrival_rate), Arrival(0))

    def schedule_arrival(self, job_id):  # TODO: complete this method
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate * self.n), Arrival(job_id))

    def schedule_completion(self, job_id):  # TODO: complete this method
        # schedule the time of the completion event
        self.schedule(expovariate(self.mu), Completion(job_id))

    @property
    def queue_len(self):
        return (self.running is not None) + len(self.queue)


class Arrival(Event):

    def __init__(self, job_id):
        self.queue_id = None
        self.id = job_id

    def process(self, sim: MMN):  # TODO: complete this method
        if self.id == 0:
            sim.schedule_arrival(self.id + 1)
        else:
            # set the arrival time of the job
            sim.arrivals[self.id] = sim.t
            print(sim.running)
            sim.running.append(self.id)
            # randomly select 5 arrays of sim.running (sim.running is an array and its size is greater than 5) as Selected Queues, now find the Minimum length in Selected Queues as minimum_length and print arrays of Selected Queues with this minimum_length            selected_queues = random.sample(sim.running, sim.d)

            sim_running_list = list(sim.queue[0].values())
            selected_queues = random.sample(sim_running_list, sim.d)

            minimum_length = min(map(len, selected_queues))

            ids = [i for i, queue in enumerate(selected_queues) if len(queue) == minimum_length]

            id = random.choice(ids)
            random_id = selected_queues[id]

            new_dict = {self.id: self.id}

            selected_queues[id].update(new_dict)
            sim.schedule_arrival(self.id)
            self.queue_id = random_id

            # set the arrival time of the job
            sim.arrivals[self.id] = sim.t
            # if there is no running job, assign the incoming one and schedule its completion
            if len(sim.running) < sim.n:
                sim.running = self.id
                sim.schedule_completion(self.id)
            # otherwise put the job into the queue
            # schedule the arrival of the next job
            sim.schedule_arrival(self.id + 1)


class Completion(Event):
    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMN):  # TODO: complete this method
        # print('ok')
        # exit()
        # assert sim.running is not None
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        # if the queue is not empty
        # if sim.queue:
        #     next_job = sim.queue[self.id]
        #     # next_job = sim.queue.popleft()
        #     sim.schedule_completion(next_job)

        # else:
        for i in sim.queue[0]:
            # print(sim.queue[0][i])
            # exit()
            single_queue = sim.queue[0][i]
            it = iter(single_queue.values())
            first, second = next(it), next(it)

            if single_queue[second] == self.id:
                # it = iter(sim.running.values())
                # first = next(it)

                # print(sim.running)
                # print(first)
                exit()
                sim.running.pop(self.id)
                copied_dict = sim.queue[self.id].copy()
                copied_dict.popitem()
                if self.id == copied_dict[0]:
                    sim.queue[self.id].pop(self.id)
                break

            # sim.running = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5, 0.9, 0.95, 0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    def erlang_c_cumulative(n, c, rho):
        # Calculate P_0
        P_0_inv = sum((c * rho) ** k / factorial(k) for k in range(c))
        P_0_inv += (c * rho) ** c / (factorial(c) * (1 - rho))
        P_0 = 1 / P_0_inv

        # Calculate P(N >= n)
        if n < c:
            P_n_or_more = sum((c * rho) ** k / factorial(k) * P_0 for k in range(n, c))
            P_n_or_more += (c * rho) ** c / (factorial(c) * (1 - rho)) * P_0
        else:
            P_n_or_more = ((c * rho) ** n / (c ** (n - c) * factorial(c)) * P_0) / (1 - rho)

        return P_n_or_more

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout
    styles = ['-', '--', '-.', ':']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(10, 6))

    for i, lambd_value in enumerate(args.lambd):
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t)  # Use lambd_value instead of args.lambd
        # exit()
        sim.run(args.max_t)
        # print(sim.running)

        completions = sim.completions
        # # print(sim.completions)
        # # exit()
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")
        print(f"Theoretical expectation for random server choice: {1 / (1 - lambd_value)}")

        # if args.csv is not None:
        #     with open(args.csv, 'a', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([lambd_value, args.mu, args.max_t, W])

    # exit()

    # Plotting
    theoretical_probs = []
    # for i, lambd_value in enumerate(args.lambd):
    #     rho = lambd_value / (args.n * args.mu)
    #     theoretical_probs = [erlang_c_cumulative(n, args.n, rho) for n in range(0, 16)]
    #     plt.plot(range(0, 16), theoretical_probs, styles[i % len(styles)], color=colors[i % len(colors)], label=f'Theoretical λ = {lambd_value}', linestyle='dotted')
    #
    # plt.xlabel('Queue Length')
    # plt.ylabel('Fraction of Queues with at least that size')
    # plt.title('Theoretical Queue Length')
    # plt.legend()
    # plt.grid(True)
    # plt.ylim(0, 1)
    # plt.xlim(1, 16)
    # plt.xticks(range(1, 17))
    # plt.show()


if __name__ == '__main__':
    main()
