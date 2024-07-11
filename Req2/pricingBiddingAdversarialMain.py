from Req2.biddingAdversarialMain import *
from pricing import *
import numpy as np


def testAgent(agent, T, env, title):
    agentRewards = np.zeros(T)
    env.reset()

    nCustomersArray = np.zeros(T)
    priceProbabilities = np.zeros((T, len(agentDiscretizedPrices)))
    for t in range(T):
        price_t = agent.pull_arm()
        print("price at day " + str(t) + ": " + str(price_t))
        valuation_t = np.clip(agent.getEstimatedRewardMean()[np.where(agentDiscretizedPrices == price_t)], 0, price_t)[0]
        print("valuation: " + str(valuation_t))
        nCustomers_t = biddingAdversarial(valuation_t, 1000, 1, 50, False, t)
        nCustomersArray[t] = nCustomers_t
        print("nCustomers: " + str(nCustomers_t))
        demand_t, reward_t = env.round(price_t, nCustomers_t)
        if nCustomers_t == 0 or demand_t == 0:
            agent.update(0)
            agentRewards[t] = 0
        else:
            agent.update(reward_t / nCustomers_t)
            agentRewards[t] = reward_t / nCustomers_t
        priceProbabilities[t, :] = agent.x_t
        print("reward: " + str(agentRewards[t]))
        print(" ")

    # Clairvoyant
    clairvoyantPrice, clairvoyantRewards = getAdversarialClairvoyant(agentDiscretizedPrices, T, env, nCustomersArray)

    plt.plot(np.arange(T), agentRewards, ".", label="Agent rewards")
    plt.plot(np.arange(T), clairvoyantRewards, ".", label="Clairvoyant rewards")
    plt.title(title)
    plt.legend()
    plt.show()

    for i, price in enumerate(agentDiscretizedPrices):
        plt.plot(np.arange(T), priceProbabilities[:, i], ".", label="Price " + str(price))
    plt.title('price probabilities')
    plt.legend()
    plt.show()
    return np.cumsum(clairvoyantRewards - agentRewards), clairvoyantPrice


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    envSeed = 50

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    priceRange = maxPrice - minPrice
    rewardRange = priceRange/10*4

    T = 100
    numDemandChanges = T-1
    numTrials = 1

    # EXP3 agent parameters
    discretization = np.floor(1 / ((T/10) ** (-1 / 3)) + 1)
    discretization = discretization.astype(int) + 2
    envDiscretizedPrices = np.round(np.linspace(minPrice, maxPrice, discretization), 5)
    agentDiscretizedPrices = envDiscretizedPrices[1:-1]

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, envDiscretizedPrices, numDemandChanges, T, envSeed, False)
    sCP = env.sortedChangePoints

    # Execute trials and rounds
    exp3Regret = np.zeros((numTrials, T))
    swUcbRegret = np.zeros((numTrials, T))
    cumSumUcbRegret = np.zeros((numTrials, T))
    for trial in range(numTrials):
        random.seed(trial)
        np.random.seed(trial)

        # EXP3 Agent
        exp3Agent = EXP3Agent(agentDiscretizedPrices, T, np.sqrt(np.log(discretization) / (discretization * T)))

        exp3Regret[trial, :], clairvoyantPrice = testAgent(exp3Agent, T, env, "EXP3 rewards")
        print("exp3 " + str(trial + 1))

        # Price history of EXP3
        plt.plot(np.arange(T), exp3Agent.pricesHistory, ".", label="Agent Prices")
        plt.plot(np.arange(T), np.repeat(clairvoyantPrice, T), label="Clairvoyant Prices")
        plt.title("EXP3 agent prices, trial " + str(trial))
        plt.legend()
        plt.show()

        # EXP3 regret
        plt.plot(np.arange(T), exp3Regret[trial, :], label="EXP3 Average Regret")
        plt.xlabel("$t$")
        plt.ylabel("Regret")
        plt.title("EXP3 regret")
        plt.show()

    exp3AverageRegret = exp3Regret.mean(axis=0)
    exp3RegretStd = exp3Regret.std(axis=0)

    # exp3 regret
    plt.plot(np.arange(T), exp3AverageRegret, label="EXP3 Average Regret")
    plt.fill_between(np.arange(T),
                     exp3AverageRegret - exp3RegretStd / np.sqrt(numTrials),
                     exp3AverageRegret + exp3RegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.title("EXP3 regret")
    plt.show()
