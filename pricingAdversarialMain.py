from pricing import *


def testAgent(agent, T, seed, expectedClairvoyantRewards):
    np.random.seed(seed)
    agentRewards = np.zeros(T)
    for t in range(T):
        price_t = agent.pull_arm()
        demand_t, reward_t = env.round(price_t, nCustomers)
        agent.update(reward_t / nCustomers, False)
        agentRewards[t] = reward_t

    return np.cumsum(expectedClairvoyantRewards - agentRewards)


if __name__ == '__main__':
    numRounds = 1000
    numPrices = 500
    env = NonStationaryBernoulliEnvironment(numPrices, numRounds, 4, 1, True)

    # Best policy in Hindsight:
    best_rewards = np.array(env.mu).max(axis=0)  # we take the max over every single round
    best_cum_rew = sum(best_rewards)
    best_policy = np.array(env.mu).argmax(axis=0)
    print(f'Best possible cumulative reward: {best_cum_rew}')  # is higher than the cumulative reward of the best arm in hindsight

    t = np.arange(numRounds)
    for i in range(numPrices):
        plt.plot(t, env.mu[i], label=f'$\mu_{i}$')
    plt.plot(t, best_rewards, label=f'$\mu^*$')

    plt.legend()
    plt.xlabel('$t$')
    plt.show()

    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    discretization = 1000
    discretizedPrices = np.linspace(minPrice, maxPrice, discretization)
    conversionProbability = lambda p: 1 - p / maxPrice
    reward_function = lambda price, n_sales: (price - cost) * n_sales

    T = 1000
    numTrials = 10
    env = StochasticEnvironment(conversionProbability, cost)

    # clairvoyant
    profit_curve = reward_function(discretizedPrices, nCustomers * conversionProbability(discretizedPrices))
    best_price_index = np.argmax(profit_curve)
    best_price = discretizedPrices[best_price_index]
    expectedClairvoyantRewards = np.repeat(profit_curve[best_price_index], T)

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        ucbAgent = GPUCBAgent(T, discretization, minPrice, maxPrice)
        ucbRegretPerTrial[trial, :] = testAgent(ucbAgent, T, trial, expectedClairvoyantRewards)

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        tsAgent = GPTSAgent(T, discretization, minPrice, maxPrice)
        tsRegretPerTrial[trial, :] = testAgent(tsAgent, T, trial, expectedClairvoyantRewards)

    clairvoyantRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        clairvoyantAgent = ClairvoyantAgent(best_price)
        clairvoyantRegretPerTrial[trial, :] = testAgent(clairvoyantAgent, T, trial, expectedClairvoyantRewards)

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
