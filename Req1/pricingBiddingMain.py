from pricing import *
from Req1.biddingMain import *
import matplotlib.ticker as mticker


def testAgent(agent, T, seed):
    np.random.seed(seed)
    agentRewards = np.zeros(T)

    for t in range(T):
        price_t = agent.pull_arm()
        print("Price at day " + str(t) + ": " + str(price_t))
        valuation_t = np.clip(agent.getEstimatedRewardMean()[np.where(discretizedPrices == price_t)], 0, price_t)
        print("Valuation: " + str(valuation_t))
        nCustomers_t = bidding(valuation_t, 0.1, 1000, 1, 50, False, t)
        print("Bids won: " + str(nCustomers_t))
        demand_t, reward_t = env.round([price_t], nCustomers_t)
        if nCustomers_t == 0 or demand_t == 0:
            agent.update(0, False)
            agentRewards[t] = 0
        else:
            agent.update(reward_t / nCustomers_t, False)
            agentRewards[t] = reward_t / nCustomers_t
        print("Reward: " + str(agentRewards[t]) + "\n")

    # clairvoyant
    expectedClairvoyantRewards = np.max(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))

    return np.cumsum(expectedClairvoyantRewards - agentRewards)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    envSeed = 50

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    priceRange = maxPrice - minPrice
    discretization = 1000
    discretizedPrices = np.linspace(minPrice, maxPrice, discretization)
    conversionProbability = lambda p: 1 - p / maxPrice
    reward_function = lambda price, n_sales: (price - cost) * n_sales

    T = 20
    numTrials = 1
    env = StochasticEnvironment([conversionProbability], [cost], envSeed)

    # clairvoyant
    bestPriceIndex = np.argmax(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))
    bestPrice = discretizedPrices[bestPriceIndex]
    print("Best price: " + str(bestPrice))

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nGPUCB trial " + str(trial + 1))
        ucbAgent = GPUCBAgent(T, discretization, [minPrice], [maxPrice])
        ucbRegretPerTrial[trial, :] = testAgent(ucbAgent, T, trial)

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nGPTS trial " + str(trial + 1))
        tsAgent = GPTSAgent(T, discretization, [minPrice], [maxPrice])
        tsRegretPerTrial[trial, :] = testAgent(tsAgent, T, trial)

    clairvoyantRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("\nBest Price trial " + str(trial + 1))
        clairvoyantAgent = BestPriceAgent(bestPrice, discretizedPrices, conversionProbability)
        clairvoyantRegretPerTrial[trial, :] = testAgent(clairvoyantAgent, T, trial)

    ucbAverageRegret = ucbRegretPerTrial.mean(axis=0)
    ucbRegretStd = ucbRegretPerTrial.std(axis=0)

    tsAverageRegret = tsRegretPerTrial.mean(axis=0)
    tsRegretStd = tsRegretPerTrial.std(axis=0)

    clairvoyantAverageRegret = clairvoyantRegretPerTrial.mean(axis=0)
    clairvoyantRegretStd = clairvoyantRegretPerTrial.std(axis=0)

    plt.plot(np.arange(T), ucbAverageRegret, label='UCB Average Regret')
    plt.plot(np.arange(T), tsAverageRegret, label='TS Average Regret')
    plt.plot(np.arange(T), clairvoyantAverageRegret, label='Best Price Average Regret')
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
                     clairvoyantAverageRegret - clairvoyantRegretStd / np.sqrt(numTrials),
                     clairvoyantAverageRegret + clairvoyantRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.ylabel('Regret')
    plt.xlabel('$t$')
    plt.legend()
    plt.show()
