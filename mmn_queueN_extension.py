#!/usr/bin/env python3

import argparse  # Importing the argparse module for parsing command-line arguments
import collections  # Importing the collections module for specialized container datatypes
import logging  # Importing the logging module for logging status messages
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for plotting graphs
from random import expovariate, seed  # Importing functions from the random module for generating random numbers
import random  # Importing the random module for random number generation
from discrete_event_sim import Simulation, Event  # Importing classes from the discrete_event_sim module


class MMN(Simulation):

    def __init__(self, lambd, mu, n, d, max_t):
        if n < 1 or d > n:
            raise NotImplementedError  # Raise an error if conditions are not met

        super().__init__()  # Call the constructor of the parent class
        self.d = d  # Number of servers
        self.queues = [collections.deque() for _ in range(n)]  # Initialize queues for each server
        self.times = []  # List to store queue lengths over time
        self.running = [None] * n  # List to keep track of running jobs on each server
        self.arrivals = {}  # Dictionary mapping job id to arrival time
        self.completions = {}  # Dictionary mapping job id to completion time
        self.lambd = lambd  # Arrival rate lambda
        self.n = n  # Total number of servers
        self.mu = mu  # Service rate mu
        self.max_t = max_t  # Maximum simulation time
        self.arrival_rate = lambd * n  # Total arrival rate to all servers
        self.array_len = [0] * self.n  # Array to store queue lengths for each server
        self.interval = 10  # Time interval for saving queue lengths
        self.schedule(0, SavingTime())  # Schedule the first event
        self.jobs = []  # List to store jobs
        self.copied_jobs = []  # Copy of the jobs list for scheduling arrivals

        # Generate initial jobs with exponential service times
        for job_id in range(0, 100000):
            self.jobs.append(job_id)
            t = expovariate(self.mu)
            self.jobs[job_id] = (job_id, t)
        self.jobs.sort(key=lambda x: x[1])  # Sort jobs based on service times
        self.copied_jobs = self.jobs.copy()  # Make a copy of the jobs list
        first_job = self.jobs.pop(0)  # Pop the first job from the list
        self.schedule(expovariate(self.arrival_rate), Arrival(first_job[0], first_job[1], 0))  # Schedule the first arrival

    # Method to schedule arrival event
    def schedule_arrival(self, job_id, mu_time):
        index = self.super_market()  # Determine the server index using the supermarket algorithm
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id, mu_time, index))  # Schedule the arrival event

    # Method to schedule completion event
    def schedule_completion(self, job_id, mu_time, index):
        self.schedule(mu_time, Completion(job_id, index))

    # Supermarket algorithm to select the server index
    def super_market(self):
        sample = random.sample(self.queues, self.d)  # Randomly select d queues
        shortest_lists = [lst for lst in sample if len(lst) == min(len(sublist) for sublist in sample)]  # Find the shortest queues
        selected_list = random.choice(shortest_lists)  # Randomly select one of the shortest queues
        index = self.queues.index(selected_list)  # Determine the index of the selected queue
        return index  # Return the index of the selected queue

    # Method to get the length of a queue
    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queues[i])  # Return the length of the queue i


class SavingTime(Event):

    # Method to process the SavingTime event
    def process(self, sim: MMN):
        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))  # Append the length of each queue to the times list
        sim.schedule(sim.interval, SavingTime())  # Schedule the next SavingTime event


class Arrival(Event):

    # Constructor for the Arrival event
    def __init__(self, job_id, mu_time, index):
        self.id = job_id  # Job ID
        self.index = index  # Server index
        self.mu_time = mu_time  # Service time

    # Method to process the Arrival event
    def process(self, sim: MMN):
        sim.arrivals[self.id] = sim.t  # Record the arrival time of the job
        if sim.running[self.index] is None:
            sim.running[self.index] = self.id
            sim.schedule_completion(self.id, self.mu_time, self.index)
        else:
            sim.queues[self.index].append({'0': self.id, '1': self.mu_time})  # Add the job to the queue
        if sim.jobs:
            job = sim.jobs.pop(0)  # Pop the next job from the jobs list
            sim.schedule_arrival(job[0], job[1])  # Schedule the arrival of the next job


class Completion(Event):

    # Constructor for the Completion event
    def __init__(self, job_id, index):
        self.id = job_id  # Job ID
        self.index = index  # Server index

    # Method to process the Completion event
    def process(self, sim: MMN):
        assert sim.running[self.index] is not None  # Ensure there is a running job on the server
        sim.completions[self.id] = sim.t  # Record the completion time of the job
        sim.running[self.index] = None  # Mark the server as idle
        if sim.queues[self.index]:
            popped_job = sim.queues[self.index].popleft()  # Remove the next job from the queue
            sim.running[self.index] = popped_job.pop('0')  # Mark the next job as running on the server
            sim.schedule_completion(sim.running[self.index], popped_job.pop('1'), self.index)  # Schedule the completion of the next job


def main():
    parser = argparse.ArgumentParser()  # Create an ArgumentParser object
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5, 0.9, 0.95, 0.99])  # Define command-line arguments for arrival rates
    parser.add_argument('--mu', type=float, default=1)  # Define command-line argument for service rate
    parser.add_argument('--max-t', type=float, default=1_000)  # Define command-line argument for maximum simulation time
    parser.add_argument('--n', type=int, default=100)  # Define command-line argument for number of servers
    parser.add_argument('--d', type=int, default=10)  # Define command-line argument for number of servers to consider in the supermarket algorithm
    parser.add_argument('--csv', help="CSV file in which to store results")  # Define command-line argument for CSV file to store results
    parser.add_argument("--seed", help="random seed")  # Define command-line argument for setting random seed
    parser.add_argument("--verbose", action='store_true')  # Define command-line argument for enabling verbose mode
    args = parser.parse_args()  # Parse the command-line arguments

    if args.seed:
        random.seed(args.seed)  # Set a random seed if provided
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # Enable logging in verbose mode

    styles = ['-', '--', '-.', ':']  # Line styles for plotting
    colors = ['blue', 'orange', 'green', 'red']  # Line colors for plotting
    plt.figure(figsize=(10, 6))  # Create a new figure for plotting

    for i, lambd_value in enumerate(args.lambd):  # Iterate over the specified lambda values
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t)  # Create an instance of the MMN simulation
        sim.run(args.max_t)  # Run the simulation with the specified maximum time

        completions = sim.completions  # Get the completion times of jobs

        # Calculate and print the average time spent in the system
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")

        # Calculate and print the theoretical expectation for random server choice
        print(f"Theoretical expectation for random server choice: {1 / (1 - lambd_value)}")

        counts = [0] * 15  # Initialize a list to store counts of queue lengths
        queueLengths = sim.times  # Get the recorded queue lengths over time

        # Count the occurrences of each queue length
        for length in queueLengths:
            if length == 0:  # Skip over queue lengths of zero
                continue
            for t in range(min(length, 15)):
                counts[t] += 1

        fractions = [count / len(queueLengths) for count in counts]  # Calculate fractions of queues with at least that size

        # Plot the theoretical queue length distribution
        plt.plot(range(1, 15), fractions[1:], color=colors[i % len(colors)], label=f'Theoretical λ = {lambd_value}',
                 linestyle=styles[i % len(styles)])

    # Set plot labels, title, legend, and limits
    plt.xlabel('Queues Length')
    plt.ylabel('Fraction of Queues with at least that size')
    plt.title(f"Theoretical Queues Length | n: {args.n} | d: {args.d} | μ: {args.mu}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, 16)
    plt.xticks(range(0, 16))
    plt.show()  # Display the plot


if __name__ == '__main__':
    main()  # Call the main function if the script is executed directly
