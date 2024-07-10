from pricing import *
import numpy as np

def getAdversarialClairvoyant(discretizedPrices, T, env):
    rewards = np.zeros((len(discretizedPrices), T))

    for i, price in enumerate(discretizedPrices):
        env.reset()
        for t in range(T):
            demand_t, rewards[i, t] = env.round(price, nCustomers)

    env.reset()
    bestPriceIndex = np.argmax(rewards.sum(axis=1))
    return discretizedPrices[bestPriceIndex], rewards[bestPriceIndex, :]


def testAgent(agent, T, clairvoyantRewards, nCustomers, env, title):
    agentRewards = np.zeros(T)
    env.reset()

    for t in range(T):
        price_t = agent.pull_arm()
        demand_t, reward_t = env.round(price_t, nCustomers)
        agent.update(reward_t / nCustomers)
        agentRewards[t] = reward_t

    plt.plot(np.arange(T), agentRewards, ".", label="Agent rewards")
    plt.plot(np.arange(T), clairvoyantRewards, ".", label="Clairvoyant rewards")
    plt.title(title)
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
    numTrials = 2

    # EXP3 agent parameters
    discretization = np.floor(1 / ((T/10) ** (-1 / 3)) + 1)
    discretization = discretization.astype(int) + 2
    discretizedPrices = np.round(np.linspace(minPrice, maxPrice, discretization), 5)
    discretizedPrices = discretizedPrices[1:-1]

    # Environment
    env = NonStationaryBernoulliEnvironment(cost, minPrice, maxPrice, discretizedPrices, numDemandChanges, T, envSeed, True)
    sCP = env.sortedChangePoints

    # Execute trials and rounds
    exp3Regret = np.zeros((numTrials, T))
    swUcbRegret = np.zeros((numTrials, T))
    cumSumUcbRegret = np.zeros((numTrials, T))
    for trial in range(numTrials):
        random.seed(trial)
        np.random.seed(trial)

        # EXP3 Agent
        exp3Agent = EXP3Agent(discretizedPrices, T, np.sqrt(np.log(discretization) / (discretization * T)))

        # Clairvoyant
        clairvoyantPrice, clairvoyantRewards = getAdversarialClairvoyant(discretizedPrices, T, env)

        exp3Regret[trial, :] = testAgent(exp3Agent, T, clairvoyantRewards, nCustomers, env, "EXP3 rewards")
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
    for x in sCP:
        plt.axvline(x, linestyle='--', color='g')
    plt.xlabel("$t$")
    plt.ylabel("Regret")
    plt.title("EXP3 regret")
    plt.show()
