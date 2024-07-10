import matplotlib.pyplot as plt

from pricing import *
import numpy as np

def testEnv(env, numTests, discretizedPrices):
    profit = np.zeros((numTests, len(discretizedPrices)))
    for n in range(numTests):
        for i, price in enumerate(discretizedPrices):
            foo, profitTemp = env.round(price, nCustomers)
            profit[n, i] = profitTemp[0]
    profitMean = profit.mean(axis=0)
    return profitMean

def testAgent(agent, T, seed, clairvoyantRewards, nCustomers, title):
    np.random.seed(seed)
    random.seed(seed)
    agentRewards = np.zeros(T)
    for t in range(T):
        price_t = agent.pull_arm()
        np.random.seed(t)
        random.seed(t)
        demand_t, reward_t = env.round(price_t, nCustomers)
        np.random.seed(seed)
        random.seed(seed)
        agent.update(reward_t / nCustomers)
        agentRewards[t] = reward_t

    plt.plot(np.arange(T), agentRewards, label="Agent rewards")
    plt.plot(np.arange(T), clairvoyantRewards, label="Clairvoyant rewards")
    if hasattr(agent, "resetTimesHistory"):
        for x in cumSumUcbAgent.resetTimesHistory:
            plt.axvline(x, linestyle='--', color='r')
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.title(title)
    plt.legend()
    plt.show()
    return np.cumsum(clairvoyantRewards - agentRewards)

def getClairvoyantRewards(agentDiscetizedPrices, env):
    discretizedMuInd = np.linspace(0, env.K-1, agentDiscetizedPrices.size+2)[1:-1].astype(int)
    mu = env.mu[:, discretizedMuInd]
    rewards = np.array(np.matmul(mu, np.diag(agentDiscetizedPrices - cost)) * nCustomers).max(axis=1)
    bestArms = np.array(np.matmul(mu, np.diag(agentDiscetizedPrices - cost)) * nCustomers).argmax(axis=1)
    bestPrices = np.array([agentDiscetizedPrices[t] for t in bestArms])
    # profitMean = testEnv(env, 10000, agentDiscetizedPrices)
    return rewards, bestPrices


if __name__ == '__main__':
    envSeed = 2
    np.random.seed(envSeed)
    random.seed(envSeed)
    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    rewardRange = priceRange/10*4

    T = 1000
    numDemandChanges = 5
    numTrials = 2

    # UCB1 agent parameters
    ucb1Discretization = np.floor(1 / (T ** (-1 / 3)) + 1)
    ucb1Discretization = ucb1Discretization.astype(int) + 2
    ucb1DiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, ucb1Discretization), 5)
    ucb1DiscretizedPrices = ucb1DiscretizedPrices[1:-1]

    # The discretization choice depends on the used learning algorithm. In case of SWUCB we use W instead of T
    W = int(2 * np.sqrt(T * np.log(T) / numDemandChanges))  # Optimal W size assuming numDemandChanges is known (otherwise sqrt(T))
    swDiscretization = np.floor(1 / (W ** (-1 / 3)) + 1)
    swDiscretization = swDiscretization.astype(int) + 2
    swDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, swDiscretization), 5)
    swDiscretizedPrices = swDiscretizedPrices[1:-1]

    # In case of CUMSUM UCB we use T/numDemandChanges
    h = 2 * np.log(T / numDemandChanges)                # Sensitivity of detection, threshold for cumulative deviation
    M = int(np.log(T / numDemandChanges))               # Robustness of change detection
    alpha = np.sqrt(numDemandChanges * np.log(T / numDemandChanges) / T)              # Probability of extra exploration
    cumSumDiscretization = np.floor(1 / ((T / numDemandChanges) ** (-1 / 3)) + 1)
    cumSumDiscretization = cumSumDiscretization.astype(int) + 2
    cumSumDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, cumSumDiscretization), 5)
    cumSumDiscretizedPrices = cumSumDiscretizedPrices[1:-1]

    # For the environment and clairvoyant, we use the l.c.m. to have a discretization which encapsulates both SW and CUMSUM
    envDiscretization = np.lcm.reduce([ucb1Discretization - 1, swDiscretization - 1, cumSumDiscretization - 1]) + 1
    envDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, envDiscretization), 5)

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
    sCP = env.sortedChangePoints

    # Clairvoyant for comparison : Best policy in Hindsight (Clairvoyant for non-stationary environment):
    clairvoyantRewards = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost)) * nCustomers).max(axis=1)  # we take the max over every single round
    clairvoyantPolicy = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost))).argmax(axis=1)

    # Clairvoyants for each algorithm:

    # Execute trials and rounds
    ucb1Regret = np.zeros((numTrials, T))
    swUcbRegret = np.zeros((numTrials, T))
    cumSumUcbRegret = np.zeros((numTrials, T))
    for trial in range(numTrials):
        # UCB1 Agent for comparison
        env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
        ucb1Agent = SWUCBAgent(ucb1DiscretizedPrices, T, T, range=rewardRange)
        ucb1ClairvoyantRewards, ucb1ClairvoyantPrices = getClairvoyantRewards(ucb1DiscretizedPrices, env)
        ucb1Regret[trial, :] = testAgent(ucb1Agent, T, trial, ucb1ClairvoyantRewards, nCustomers, "UCB1 rewards")
        print("ucb1 " + str(trial + 1))

        # Price history of UCB1
        plt.plot(np.arange(T), ucb1Agent.pricesHistory, label="Agent Prices")
        plt.plot(np.arange(T), ucb1ClairvoyantPrices, label="Clairvoyant Prices")
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.title("UCB1 agent prices, trial " + str(trial))
        plt.legend()
        plt.show()

        # UCBs history of UCB1
        for x in range(ucb1Agent.K) : plt.plot(np.arange(T), ucb1Agent.ucbsHistory[:,x])
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.title("UCB1 ucbs history, trial" + str(trial))
        plt.show()

        # UCB1 regret
        plt.plot(np.arange(T), ucb1Regret[trial, :], label="SW UCB Average Regret")
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.xlabel("$t$")
        plt.ylabel("Regret")
        plt.title("UCB1 regret")
        plt.show()

        # SW UCB Agent
        env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
        swUcbAgent = SWUCBAgent(swDiscretizedPrices, T, W, range=rewardRange)
        swClairvoyantRewards, swClairvoyantPrices = getClairvoyantRewards(swDiscretizedPrices, env)
        swUcbRegret[trial, :] = testAgent(swUcbAgent, T, trial, swClairvoyantRewards, nCustomers, "SW UCB rewards")
        print("swUcb " + str(trial + 1))

        # Price history of SW UCB
        plt.plot(np.arange(T), swUcbAgent.pricesHistory, label="Agent Prices")
        plt.plot(np.arange(T), swClairvoyantPrices, label="Clairvoyant Prices")
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.title("SW UCB agent prices, trial " + str(trial))
        plt.legend()
        plt.show()

        # UCBs history of SW UCB
        for x in range(swUcbAgent.K): plt.plot(np.arange(T), swUcbAgent.ucbsHistory[:, x])
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.title("SW UCB ucbs history, trial" + str(trial))
        plt.show()

        # Regret of SW UCB
        plt.plot(np.arange(T), swUcbRegret[trial, :], label="SW UCB Average Regret")
        for x in range(0, T + 1, W):
            plt.axvline(x, linestyle='--', color='r')
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.xlabel("$t$")
        plt.ylabel("Regret")
        plt.title("SW UCB regret, trial = " + str(trial))
        plt.show()

        # CumSum UCB Agent
        env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
        cumSumUcbAgent = CUSUMUCBAgent(cumSumDiscretizedPrices, T, M, h, range=rewardRange)
        cumSumClairvoyantRewards, cumSumClairvoyantPrices = getClairvoyantRewards(cumSumDiscretizedPrices, env)
        cumSumUcbRegret[trial, :] = testAgent(cumSumUcbAgent, T, trial, cumSumClairvoyantRewards, nCustomers, "CUMSUM UCB rewards")
        print("cumSumUcb " + str(trial + 1))

        # Price history of CUMSUM UCB
        plt.plot(np.arange(T), cumSumUcbAgent.pricesHistory, label="Agent Prices")
        plt.plot(np.arange(T), cumSumClairvoyantPrices, label="Clairvoyant Prices")
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        for x in cumSumUcbAgent.resetTimesHistory:
            plt.axvline(x, linestyle='--', color='r')
        plt.title("CUMSUM agent prices, trial " + str(trial))
        plt.show()

        # UCBs history of CUMSUM UCB
        for x in range(cumSumUcbAgent.K): plt.plot(np.arange(T), cumSumUcbAgent.ucbsHistory[:, x])
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        for x in cumSumUcbAgent.resetTimesHistory:
            plt.axvline(x, linestyle='--', color='r')
        plt.title("CUMSUM UCB ucbs history, trial" + str(trial))
        plt.show()

        # CumSum with detected changes
        plt.plot(np.arange(T), cumSumUcbRegret[trial, :], label="CUMSUM UCB Average Regret")
        for x in cumSumUcbAgent.resetTimesHistory:
            plt.axvline(x, linestyle='--', color='r')
        for x in sCP:
            plt.axvline(x, linestyle='--', color='g')
        plt.xlabel("$t$")
        plt.ylabel("Regret")
        plt.title("CUMSUM UCB regret")
        plt.show()

    ucb1AverageRegret = ucb1Regret.mean(axis=0)
    ucb1RegretStd = ucb1Regret.std(axis=0)

    swUcbAverageRegret = swUcbRegret.mean(axis=0)
    swUcbRegretStd = swUcbRegret.std(axis=0)

    cumSumUcbAverageRegret = cumSumUcbRegret.mean(axis=0)
    cumSumUcbRegretStd = cumSumUcbRegret.std(axis=0)

    '''
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

    # Ucb1 regret
    plt.plot(np.arange(T), ucb1AverageRegret, label="SW UCB Average Regret")
    plt.fill_between(np.arange(T),
                     ucb1AverageRegret - ucb1RegretStd / np.sqrt(numTrials),
                     ucb1AverageRegret + ucb1RegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.title("UCB1 regret")
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
    plt.title("SW UCB regret")
    plt.show()
    '''



