from pricing import *
import numpy as np
import random


def testEnv(env, numTests, discretizedPrices):
    profit = np.zeros((numTests, len(discretizedPrices)))
    for n in range(numTests):
        for i, price in enumerate(discretizedPrices):
            foo, profitTemp = env.round(price, nCustomers)
            profit[n, i] = profitTemp[0]
    profitMean = profit.mean(axis=0)
    return profitMean


def testAgent(env, agent, T, specificClairvoyantRewards, generalClairvoyantRewards, nCustomers, title):
    env.reset()
    agentRewards = np.zeros(T)
    for t in range(T):
        price_t = agent.pull_arm()
        demand_t, reward_t = env.round(price_t, nCustomers)
        agent.update(reward_t / nCustomers)
        agentRewards[t] = reward_t
    '''
    plt.plot(np.arange(T), agentRewards, ".", label="Agent rewards")
    plt.plot(np.arange(T), specificClairvoyantRewards, label="Clairvoyant rewards")
    if hasattr(agent, "resetTimesHistory"):
        for x in cumSumUcbAgent.resetTimesHistory:
            plt.axvline(x, linestyle='--', color='r')
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.title(title)
    plt.legend()
    plt.show()
    '''
    return np.cumsum(specificClairvoyantRewards - agentRewards), np.cumsum(generalClairvoyantRewards - agentRewards)


def getClairvoyantRewards(agentDiscetizedPrices, env):
    discretizedMuInd = np.linspace(0, env.K-1, agentDiscetizedPrices.size+2)[1:-1].astype(int)
    mu = env.mu[:, discretizedMuInd]
    rewards = np.array(np.matmul(mu, np.diag(agentDiscetizedPrices - cost)) * nCustomers).max(axis=1)
    bestArms = np.array(np.matmul(mu, np.diag(agentDiscetizedPrices - cost)) * nCustomers).argmax(axis=1)
    bestPrices = np.array([agentDiscetizedPrices[t] for t in bestArms])
    # profitMean = testEnv(env, 10000, agentDiscetizedPrices)
    return rewards, bestPrices


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    envSeed = 50

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    rewardRange = priceRange/10

    T = 1000
    numDemandChanges = 5
    numTrials = 1

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
    h = 0.5 * np.log(T / numDemandChanges)                # Sensitivity of detection, threshold for cumulative deviation
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
    generalClairvoyantRewards = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost)) * nCustomers).max(axis=1)  # we take the max over every single round
    generalClairvoyantPolicy = np.array(np.matmul(env.mu, np.diag(envDiscretizedPrices - cost))).argmax(axis=1)

    # Execute trials and rounds
    ucb1Regret = np.zeros((numTrials, T))
    ucb1GeneralRegret = np.zeros((numTrials, T))
    swUcbRegret = np.zeros((numTrials, T))
    swUcbGeneralRegret = np.zeros((numTrials, T))
    cumSumUcbRegret = np.zeros((numTrials, T))
    cumSumUcbGeneralRegret = np.zeros((numTrials, T))
    for trial in range(numTrials):
        np.random.seed(trial)
        random.seed(trial)

        # UCB1 Agent for comparison
        print("UCB1 trial " + str(trial + 1))
        ucb1Agent = SWUCBAgent(ucb1DiscretizedPrices, T, T, range=rewardRange)
        ucb1ClairvoyantRewards, ucb1ClairvoyantPrices = getClairvoyantRewards(ucb1DiscretizedPrices, env)
        ucb1Regret[trial, :], ucb1GeneralRegret[trial, :] = testAgent(env, ucb1Agent, T, ucb1ClairvoyantRewards, generalClairvoyantRewards, nCustomers, "UCB1 rewards")
        '''
        # Price history of UCB1
        plt.plot(np.arange(T), ucb1Agent.pricesHistory, ".", label="Agent Prices")
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
        '''

        # SW UCB Agent
        print("SWUCB trial " + str(trial + 1))
        swUcbAgent = SWUCBAgent(swDiscretizedPrices, T, W, range=rewardRange)
        swClairvoyantRewards, swClairvoyantPrices = getClairvoyantRewards(swDiscretizedPrices, env)
        swUcbRegret[trial, :], swUcbGeneralRegret[trial, :] = testAgent(env, swUcbAgent, T, swClairvoyantRewards, generalClairvoyantRewards, nCustomers, "SW UCB rewards")
        '''
        # Price history of SW UCB
        plt.plot(np.arange(T), swUcbAgent.pricesHistory, ".", label="Agent Prices")
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
        '''

        # CumSum UCB Agent
        print("CUMSUM trial " + str(trial + 1))
        cumSumUcbAgent = CUSUMUCBAgent(cumSumDiscretizedPrices, T, M, h, range=rewardRange)
        cumSumClairvoyantRewards, cumSumClairvoyantPrices = getClairvoyantRewards(cumSumDiscretizedPrices, env)
        cumSumUcbRegret[trial, :], cumSumUcbGeneralRegret[trial, :] = testAgent(env, cumSumUcbAgent, T, cumSumClairvoyantRewards, generalClairvoyantRewards, nCustomers, "CUMSUM UCB rewards")
        '''
        # Price history of CUMSUM UCB
        plt.plot(np.arange(T), cumSumUcbAgent.pricesHistory, ".", label="Agent Prices")
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
        '''

    ucb1SpecificAvRegret = ucb1Regret.mean(axis=0)
    ucb1SpecRegretStd = ucb1Regret.std(axis=0)
    ucb1GeneralAvRegret = ucb1GeneralRegret.mean(axis=0)
    ucb1GeneralRegretStd = ucb1GeneralRegret.std(axis=0)

    swUcbAverageRegret = swUcbRegret.mean(axis=0)
    swUcbRegretStd = swUcbRegret.std(axis=0)
    swUcbGeneralAvRegret = swUcbGeneralRegret.mean(axis=0)
    swUcbGeneralRegretStd = swUcbGeneralRegret.std(axis=0)

    cumSumUcbAverageRegret = cumSumUcbRegret.mean(axis=0)
    cumSumUcbRegretStd = cumSumUcbRegret.std(axis=0)
    cumSumUcbGeneralAvRegret = cumSumUcbGeneralRegret.mean(axis=0)
    cumSumUcbGeneralRegretStd = cumSumUcbGeneralRegret.std(axis=0)

    # Ucb1 regret
    plt.plot(np.arange(T), ucb1SpecificAvRegret, label="SW UCB Average Regret")
    plt.fill_between(np.arange(T),
                     ucb1SpecificAvRegret - ucb1SpecRegretStd / np.sqrt(numTrials),
                     ucb1SpecificAvRegret + ucb1SpecRegretStd / np.sqrt(numTrials),
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

    # CumSum Agent regret
    plt.plot(np.arange(T), cumSumUcbAverageRegret, label="CUMSUM UCB Average Regret")
    plt.fill_between(np.arange(T),
                     cumSumUcbAverageRegret - cumSumUcbRegretStd / np.sqrt(numTrials),
                     cumSumUcbAverageRegret + cumSumUcbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    for x in cumSumUcbAgent.resetTimesHistory:
        plt.axvline(x, linestyle='--', color='r')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.title("CUMSUM UCB regret")
    plt.show()

    # Comparison graph
    plt.plot(np.arange(T), ucb1GeneralAvRegret, label="UCB1 Average Regret")
    plt.fill_between(np.arange(T),
                     ucb1GeneralAvRegret - ucb1GeneralRegretStd / np.sqrt(numTrials),
                     ucb1GeneralAvRegret + ucb1GeneralRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.plot(np.arange(T), swUcbGeneralAvRegret, label="SW UCB Average Regret")
    plt.fill_between(np.arange(T),
                     swUcbGeneralAvRegret - swUcbGeneralRegretStd / np.sqrt(numTrials),
                     swUcbGeneralAvRegret + swUcbGeneralRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.plot(np.arange(T), cumSumUcbGeneralAvRegret, label="CUMSUM UCB Average Regret")
    plt.fill_between(np.arange(T),
                     cumSumUcbGeneralAvRegret - cumSumUcbGeneralRegretStd / np.sqrt(numTrials),
                     cumSumUcbGeneralAvRegret + cumSumUcbGeneralRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
