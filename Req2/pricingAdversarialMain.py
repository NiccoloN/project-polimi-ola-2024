from pricing import *
import random
import numpy as np


def testAgent(agent, T, clairvoyantRewards, nCustomers, env, title):
    agentRewards = np.zeros(T)
    env.reset()

    for t in range(T):
        price_t = agent.pull_arm()
        demand_t, reward_t = env.round(price_t, nCustomers)
        agent.update(reward_t / nCustomers)
        agentRewards[t] = reward_t / nCustomers

    plt.plot(np.arange(T), agentRewards, ".", label="EXP3 Rewards")
    plt.plot(np.arange(T), clairvoyantRewards, ".", label="Clairvoyant rewards")
    plt.title(title)
    plt.ylabel("$Reward$")
    plt.xlabel("$t$")
    plt.legend()
    plt.show()
    return np.cumsum(clairvoyantRewards - agentRewards)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    envSeed = 50

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    rewardRange = priceRange/10*4

    T = 10000
    numDemandChanges = T-1
    numTrials = 3

    # EXP3 agent parameters
    discretization = np.floor(1 / ((T/10) ** (-1 / 3)) + 1)
    discretization = discretization.astype(int) + 2
    envDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, discretization), 5)
    agentDiscretizedPrices = envDiscretizedPrices[1:-1]

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
    sCP = env.sortedChangePoints

    # Clairvoyant
    clairvoyantPrice, clairvoyantRewards = getAdversarialClairvoyant(agentDiscretizedPrices, T, env, np.repeat(nCustomers, T))
    print("Best price: " + str(clairvoyantPrice) + "\n")

    # Execute trials
    exp3Regret = np.zeros((numTrials, T))
    for trial in range(numTrials):
        random.seed(trial)
        np.random.seed(trial)

        # EXP3 Agent
        exp3Agent = EXP3Agent(agentDiscretizedPrices, T, np.sqrt(np.log(discretization) / (discretization * T)))

        exp3Regret[trial, :] = testAgent(exp3Agent, T, clairvoyantRewards, nCustomers, env, "EXP3 rewards")
        print("EXP3 trial " + str(trial + 1))

        # Price history of EXP3
        plt.plot(np.arange(T), exp3Agent.pricesHistory, ".", label="EXP3 Prices")
        plt.plot(np.arange(T), np.repeat(clairvoyantPrice, T), label="Clairvoyant Prices")
        plt.title("EXP3 Agent Prices (trial " + str(trial+1) + ")")
        plt.ylabel("$Price$")
        plt.xlabel("$t$")
        plt.legend()
        plt.show()

        # EXP3 regret
        plt.plot(np.arange(T), exp3Regret[trial, :], label="EXP3 Regret")
        plt.xlabel("$t$")
        plt.ylabel("Regret")
        plt.title("EXP3 Regret (trial " + str(trial+1) + ")")
        plt.show()

    exp3AverageRegret = exp3Regret.mean(axis=0)
    exp3RegretStd = exp3Regret.std(axis=0)

    # EXP3 average regret
    plt.plot(np.arange(T), exp3AverageRegret, label="EXP3 Average Regret")
    plt.fill_between(np.arange(T),
                     exp3AverageRegret - exp3RegretStd / np.sqrt(numTrials),
                     exp3AverageRegret + exp3RegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.title("EXP3 Average Regret")
    plt.show()
