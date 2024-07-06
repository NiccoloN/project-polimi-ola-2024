import matplotlib.pyplot as plt

from pricing import *
import numpy as np


def testAgent(agent, T, seed, clairvoyantRewards, nCustomers):
    np.random.seed(seed)
    agentRewards = np.zeros(T)
    for t in range(T):
        price_t = agent.pull_arm()
        demand_t, reward_t = env.round(price_t, nCustomers)
        agent.update(reward_t / nCustomers)
        agentRewards[t] = reward_t
    return np.cumsum(clairvoyantRewards - agentRewards)


if __name__ == '__main__':
    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice

    T = 1000
    numDemandChanges = 5
    numTrials = 20

    # The discretization choice depends on the used learning algorithm. In case of SWUCB we use W instead of T
    W = int(2 * np.sqrt(T * np.log(T) / numDemandChanges))  # Optimal W size assuming numDemandChanges is known (otherwise sqrt(T))
    # W = T
    SWdiscretization = np.floor(1 / (W ** (-1/3)) + 1)
    SWdiscretization = SWdiscretization.astype(int)
    SWdiscretizedPrices = np.linspace(minPrice, maxPrice, SWdiscretization)

    # In case of CUMSUM UCB we use T/numDemandChanges
    h = 2 * np.log(T / numDemandChanges)                    # Sensitivity of detection, threshold for cumulative deviation
    M = int(np.log(T / numDemandChanges))                   # robustness of change detection
    CUMSUMdiscretization = np.floor(1 / ((T/5) ** (-1/3)) + 1)
    CUMSUMdiscretization = CUMSUMdiscretization.astype(int)
    CUMSUMdiscretizedPrices = np.linspace(minPrice, maxPrice, CUMSUMdiscretization)

    # For the environment and clairvoyant, we use the l.c.m. to have a discretization which encapsulates both SW and CUMSUM
    envDiscretization = np.lcm(SWdiscretization-1, CUMSUMdiscretization-1)+1
    envDiscretizedPrices = np.linspace(minPrice, maxPrice, envDiscretization)

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, 2, False)
    sCP = env.sortedChangePoints

    # Clairvoyant : Best policy in Hindsight (Clairvoyant for non-stationary environment):
    clairvoyantRewards = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost)) * nCustomers).max(axis=1)  # we take the max over every single round
    clairvoyantPolicy = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost))).argmax(axis=1)

    # Execute trials and rounds
    swUcbRegretPerTrial = np.zeros((numTrials, T))
    cumSumUbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):

        # SW UCB Agent
        env.t = 0   # Environment reset
        swUcbAgent = SWUCBAgent(SWdiscretizedPrices, T, W, range=priceRange/10)
        swUcbRegretPerTrial[trial, :] = testAgent(swUcbAgent, T, trial, clairvoyantRewards, nCustomers)
        print("swUcb " + str(trial + 1))

        # CumSum UCB Agent
        env.t = 0  # Environment reset
        cumSumUcbAgent = CUSUMUCBAgent(CUMSUMdiscretizedPrices, T, M, h, range=priceRange/10)
        cumSumUbRegretPerTrial[trial, :] = testAgent(cumSumUcbAgent, T, trial, clairvoyantRewards, nCustomers)
        print("sumSumUcb " + str(trial + 1))

    swUcbAverageRegret = swUcbRegretPerTrial.mean(axis=0)
    swUcbRegretStd = swUcbRegretPerTrial.std(axis=0)

    cumSumUcbAverageRegret = cumSumUbRegretPerTrial.mean(axis=0)
    cumSumUcbRegretStd = cumSumUbRegretPerTrial.std(axis=0)

    # Comparison graph
    plt.plot(np.arange(T), swUcbAverageRegret, label="SW UCB Average Regret")
    plt.fill_between(np.arange(T),
                     swUcbAverageRegret - swUcbRegretStd / np.sqrt(numTrials),
                     swUcbAverageRegret + swUcbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.plot(np.arange(T), cumSumUcbAverageRegret, label="CUMSUM UCB Average Regret")
    plt.fill_between(np.arange(T),
                     cumSumUcbAverageRegret - cumSumUcbRegretStd / np.sqrt(numTrials),
                     cumSumUcbAverageRegret + cumSumUcbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.show()

    # Sliding window multiples highlighted
    plt.plot(np.arange(T), swUcbAverageRegret, label="SW UCB Average Regret")
    plt.fill_between(np.arange(T),
                     swUcbAverageRegret - swUcbRegretStd / np.sqrt(numTrials),
                     swUcbAverageRegret + swUcbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)

    for x in range(0, T + 1, W):
        plt.axvline(x, linestyle='--', color='r')
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.show()

