import random
from pricing import *


def testAgent(env, agent, T, expectedClairvoyantRewards):
    env.reset()
    agentRewards = np.zeros(T)
    for t in range(T):
        price_t = agent.pull_arm()
        print("Price at day " + str(t) + ": " + str(price_t))
        demand_t, reward_t = env.round([price_t], nCustomers)
        agent.update(reward_t / nCustomers, False)
        agentRewards[t] = reward_t

    return np.cumsum(expectedClairvoyantRewards - agentRewards)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    envSeed = 4

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    discretization = 1000
    discretizedPrices = np.linspace(minPrice, maxPrice, discretization)
    conversionProbability = lambda p: 1 - p / maxPrice
    reward_function = lambda price, n_sales: (price - cost) * n_sales

    T = 100
    numTrials = 1
    env = StochasticEnvironment([conversionProbability], [cost], envSeed)

    # clairvoyant
    profitCurve = reward_function(discretizedPrices, nCustomers * conversionProbability(discretizedPrices))
    bestPriceIndex = np.argmax(profitCurve)
    bestPrice = discretizedPrices[bestPriceIndex]
    print("Best price: " + str(bestPrice))
    expectedClairvoyantRewards = np.repeat(profitCurve[bestPriceIndex], T)

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nGPUCB trial " + str(trial + 1))
        ucbAgent = GPUCBAgent(T, discretization, [minPrice], [maxPrice])
        ucbRegretPerTrial[trial, :] = testAgent(env, ucbAgent, T, expectedClairvoyantRewards)

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nGPTS trial " + str(trial + 1))
        tsAgent = GPTSAgent(T, discretization, [minPrice], [maxPrice])
        tsRegretPerTrial[trial, :] = testAgent(env, tsAgent, T, expectedClairvoyantRewards)

    bestPriceRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nBest price trial " + str(trial + 1))
        bestPriceAgent = BestPriceAgent(bestPrice, discretizedPrices, conversionProbability)
        bestPriceRegretPerTrial[trial, :] = testAgent(env, bestPriceAgent, T, expectedClairvoyantRewards)

    ucbAverageRegret = ucbRegretPerTrial.mean(axis=0)
    ucbRegretStd = ucbRegretPerTrial.std(axis=0)

    tsAverageRegret = tsRegretPerTrial.mean(axis=0)
    tsRegretStd = tsRegretPerTrial.std(axis=0)

    bestPriceAverageRegret = bestPriceRegretPerTrial.mean(axis=0)
    bestPriceRegretStd = bestPriceRegretPerTrial.std(axis=0)

    plt.plot(np.arange(T), ucbAverageRegret, label='UCB Average Regret')
    plt.plot(np.arange(T), tsAverageRegret, label='TS Average Regret')
    plt.plot(np.arange(T), bestPriceAverageRegret, label='Best Price Average Regret')
    plt.title('Cumulative Regret')
    plt.fill_between(np.arange(T),
                     ucbAverageRegret - ucbRegretStd / np.sqrt(numTrials),
                     ucbAverageRegret + ucbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.fill_between(np.arange(T),
                     tsAverageRegret - tsRegretStd / np.sqrt(numTrials),
                     tsAverageRegret + tsRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.fill_between(np.arange(T),
                     bestPriceAverageRegret - bestPriceRegretStd / np.sqrt(numTrials),
                     bestPriceAverageRegret + bestPriceRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.ylabel('Regret')
    plt.xlabel('$t$')
    plt.legend()
    plt.show()
