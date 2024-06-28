from pricing import *
from biddingMain import *

def testAgent(agent, T, seed):
    np.random.seed(seed)
    agentRewards = np.zeros(T)

    nCustomersArray = []
    for t in range(T):
        price_t = agent.pull_arm()
        print("price at day " + str(t) + ": " + str(price_t))
        valuation_t = price_t * conversionProbability(price_t)
        nCustomers_t = bidding(valuation_t, 1000, 1, 200, False)
        nCustomersArray.append(nCustomers_t)
        demand_t, reward_t = env.round(price_t, nCustomers_t)
        if nCustomers_t == 0:
            agent.update(0, False)
        else:
            agent.update(reward_t / nCustomers_t, False)
        agentRewards[t] = reward_t

    # clairvoyant
    best_price_index = np.argmax(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))
    expectedClairvoyantRewards = nCustomersArray * discretizedPrices[best_price_index]

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

    T = 100
    numTrials = 1
    env = StochasticEnvironment(conversionProbability, cost)

    # clairvoyant
    best_price_index = np.argmax(reward_function(discretizedPrices, conversionProbability(discretizedPrices)))
    best_price = discretizedPrices[best_price_index]

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        ucbAgent = GPUCBAgent(T, discretization, minPrice, maxPrice)
        ucbRegretPerTrial[trial, :] = testAgent(ucbAgent, T, trial)
        print("ucb " + str(trial + 1))

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        tsAgent = GPTSAgent(T, discretization, minPrice, maxPrice)
        tsRegretPerTrial[trial, :] = testAgent(tsAgent, T, trial)
        print("ts " + str(trial + 1))

    clairvoyantRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        clairvoyantAgent = ClairvoyantAgent(best_price)
        clairvoyantRegretPerTrial[trial, :] = testAgent(clairvoyantAgent, T, trial)
        print("clairvoyant " + str(trial + 1))

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
