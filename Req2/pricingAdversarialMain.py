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
    numTrials = 10

    # UCB1 agent parameters
    ucb1Discretization = np.floor(1 / (T ** (-1 / 3)) + 1)
    ucb1Discretization = ucb1Discretization.astype(int) + 2
    ucb1DiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, ucb1Discretization), 5)
    ucb1DiscretizedPrices = ucb1DiscretizedPrices[1:-1]

    # The discretization choice depends on the used learning algorithm. In case of SWUCB we use W instead of T
    W = int(2 * np.sqrt(T * np.log(T) / numDemandChanges))  # Optimal W size assuming numDemandChanges is known (otherwise sqrt(T))
    # W = T
    swDiscretization = np.floor(1 / (W ** (-1 / 3)) + 1)
    swDiscretization = swDiscretization.astype(int) + 2
    swDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, swDiscretization), 5)
    swDiscretizedPrices = swDiscretizedPrices[1:-1]


    # In case of CUMSUM UCB we use T/numDemandChanges
    h = 0.5 * np.log(T / numDemandChanges)                # Sensitivity of detection, threshold for cumulative deviation
    M = int(np.log(T / numDemandChanges))               # Robustness of change detection
    alpha = np.sqrt(numDemandChanges * np.log(T / numDemandChanges) / T)              # Probability of extra exploration
    cumSumDiscretization = np.floor(1 / ((T / 5) ** (-1 / 3)) + 1)
    cumSumDiscretization = cumSumDiscretization.astype(int) + 2
    cumSumDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, cumSumDiscretization), 5)
    cumSumDiscretizedPrices = cumSumDiscretizedPrices[1:-1]

    # For the environment and clairvoyant, we use the l.c.m. to have a discretization which encapsulates both SW and CUMSUM
    envDiscretization = np.lcm.reduce([ucb1Discretization - 1, swDiscretization - 1, cumSumDiscretization - 1]) + 1
    envDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, envDiscretization), 5)

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, 2, True)
    sCP = env.sortedChangePoints

    # Clairvoyant : Best policy in Hindsight (Clairvoyant for non-stationary environment):
    clairvoyantRewards = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost)) * nCustomers).max(axis=1)  # we take the max over every single round
    clairvoyantPolicy = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost))).argmax(axis=1)

    # Execute trials and rounds
    ucb1RegretPerTrial = np.zeros((numTrials, T))
    swUcbRegretPerTrial = np.zeros((numTrials, T))
    cumSumUbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        # UCB1 Agent for comparison
        env.t = 0  # Environment reset
        ucb1Agent = SWUCBAgent(ucb1DiscretizedPrices, T, T, range=priceRange / 4)
        ucb1RegretPerTrial[trial, :] = testAgent(ucb1Agent, T, trial, clairvoyantRewards, nCustomers)
        print("ucb1 " + str(trial + 1))

        # SW UCB Agent
        env.t = 0   # Environment reset
        swUcbAgent = SWUCBAgent(swDiscretizedPrices, T, W, range=priceRange / 4)
        swUcbRegretPerTrial[trial, :] = testAgent(swUcbAgent, T, trial, clairvoyantRewards, nCustomers)
        print("swUcb " + str(trial + 1))

        # CumSum UCB Agent
        env.t = 0  # Environment reset
        cumSumUcbAgent = CUSUMUCBAgent(cumSumDiscretizedPrices, T, M, h, range=priceRange / 4)
        cumSumUbRegretPerTrial[trial, :] = testAgent(cumSumUcbAgent, T, trial, clairvoyantRewards, nCustomers)
        print("cumSumUcb " + str(trial + 1))

    # CumSum with detected changes
    plt.plot(np.arange(T), cumSumUbRegretPerTrial[trial,:], label="CUMSUM UCB Average Regret")
    for x in cumSumUcbAgent.resetTimes:
        plt.axvline(x, linestyle='--', color='r')
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.show()

    ucb1AverageRegret = ucb1RegretPerTrial.mean(axis=0)
    ucb1RegretStd = ucb1RegretPerTrial.std(axis=0)

    swUcbAverageRegret = swUcbRegretPerTrial.mean(axis=0)
    swUcbRegretStd = swUcbRegretPerTrial.std(axis=0)

    cumSumUcbAverageRegret = cumSumUbRegretPerTrial.mean(axis=0)
    cumSumUcbRegretStd = cumSumUbRegretPerTrial.std(axis=0)

    # Comparison graph
    plt.plot(np.arange(T), ucb1AverageRegret, label="UCB1 Average Regret")
    plt.fill_between(np.arange(T),
                     ucb1AverageRegret - ucb1RegretStd / np.sqrt(numTrials),
                     ucb1AverageRegret + ucb1RegretStd / np.sqrt(numTrials),
                     alpha=0.3)
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
    plt.legend()
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


