import numpy as np

from pricing import *
from Req1.biddingMain import *

def testAgent(agent, T, seed):
    np.random.seed(seed)
    agentRewards = np.zeros(T)

    for t in range(T):
        price_t = agent.pull_arm()
        print("price at day " + str(t) + ": " + str(price_t))
        valuation_t = np.clip(agent.getEstimatedRewardMean()[np.where(discretizedPrices == price_t)], 0, price_t)
        print("valuation: " + str(valuation_t))
        nCustomers_t = bidding(valuation_t, 0.1, 1000, 1, 50, False, t)
        print("nCustomers: " + str(nCustomers_t))
        demand_t, reward_t = env.round(price_t, nCustomers_t)
        print("Est. conv probability: " + str(demand_t/nCustomers_t))
        if nCustomers_t == 0 or demand_t == 0:
            agent.update(0, False)
            agentRewards[t] = 0
        else:
            agent.update(reward_t / nCustomers_t, False)
            agentRewards[t] = reward_t / nCustomers_t
        print("reward: " + str(agentRewards[t]))
        print(" ")

    # clairvoyant
    expectedClairvoyantRewards = np.max(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))

    return np.cumsum(expectedClairvoyantRewards - agentRewards)


if __name__ == '__main__':
    cost = 0.2
    minPrice = cost
    maxPrice = 1
    priceRange = maxPrice - minPrice
    discretization = 1000
    discretizedPrices = np.linspace(minPrice, maxPrice, discretization)
    conversionProbability = lambda p: 1 - p / maxPrice
    reward_function = lambda price, n_sales: (price - cost) * n_sales

    T = 10
    numTrials = 1
    env = StochasticEnvironment(conversionProbability, cost)

    # clairvoyant
    best_price_index = np.argmax(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))
    best_price = discretizedPrices[best_price_index]

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("GPUCB agent, trial " + str(trial + 1))
        ucbAgent = GPUCBAgent(T, discretization, minPrice, maxPrice)
        ucbRegretPerTrial[trial, :] = testAgent(ucbAgent, T, trial)

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("GPTS agent, trial " + str(trial + 1))
        tsAgent = GPTSAgent(T, discretization, minPrice, maxPrice)
        tsRegretPerTrial[trial, :] = testAgent(tsAgent, T, trial)

    clairvoyantRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("clairvoyant agent, trial " + str(trial + 1))
        clairvoyantAgent = ClairvoyantAgent(best_price, discretizedPrices, conversionProbability)
        clairvoyantRegretPerTrial[trial, :] = testAgent(clairvoyantAgent, T, trial)

    ucbAverageRegret = ucbRegretPerTrial.mean(axis=0)
    ucbRegretStd = ucbRegretPerTrial.std(axis=0)

    tsAverageRegret = tsRegretPerTrial.mean(axis=0)
    tsRegretStd = tsRegretPerTrial.std(axis=0)

    clairvoyantAverageRegret = clairvoyantRegretPerTrial.mean(axis=0)
    clairvoyantRegretStd = clairvoyantRegretPerTrial.std(axis=0)

    plt.plot(np.arange(T), ucbAverageRegret, label='UCB Average Regret')
    plt.plot(np.arange(T), tsAverageRegret, label='TS Average Regret')
    plt.plot(np.arange(T), clairvoyantAverageRegret, label='Clairvoyant Average Regret')
    plt.title('cumulative regret of UCB and TS')
    plt.fill_between(np.arange(T),
                     ucbAverageRegret - ucbRegretStd / np.sqrt(numTrials),
                     ucbAverageRegret + ucbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.fill_between(np.arange(T),
                     tsAverageRegret - tsRegretStd / np.sqrt(numTrials),
                     tsAverageRegret + tsRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.fill_between(np.arange(T),
                     clairvoyantAverageRegret - clairvoyantRegretStd / np.sqrt(numTrials),
                     clairvoyantAverageRegret + clairvoyantRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.xlabel('$t$')
    plt.legend()
    plt.show()
