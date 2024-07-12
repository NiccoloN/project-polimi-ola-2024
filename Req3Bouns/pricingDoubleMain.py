import random
from pricing import *


def testAgent(env, agent, T, expectedClairvoyantRewards):
    env.reset()
    agentRewards = np.zeros(T)
    for t in range(T):
        prices_t = agent.pull_arm()
        print("Prices at day " + str(t) + ": " + str(prices_t))
        demand_t, reward_t = env.round(prices_t, nCustomers)
        agent.update(reward_t / nCustomers, False)
        agentRewards[t] = reward_t

    return np.cumsum(expectedClairvoyantRewards - agentRewards)


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    envSeed = 50

    costs = [0, 0]
    minPrices = costs
    maxPrices = np.repeat(1, 2)
    nCustomers = 100
    priceRange = maxPrices - minPrices
    discretization = 100
    discretizedPrices1 = np.linspace(minPrices[0], maxPrices[0], discretization)
    discretizedPrices2 = np.linspace(minPrices[1], maxPrices[1], discretization)
    conversionProbabilityEq = lambda x, y: math.exp(-(1.4 * x)**2) * math.log(y + 1)
    conversionProbabilityP1 = lambda p1, p2: conversionProbabilityEq(p1, p2)
    conversionProbabilityP2 = lambda p1, p2: conversionProbabilityEq(p2, p1)
    reward_function = lambda prices, n_sales: np.sum((prices - costs) * n_sales)

    T = 100
    numTrials = 1
    env = StochasticEnvironment([conversionProbabilityP1, conversionProbabilityP2], costs, envSeed)

    # clairvoyant
    profitCurve = np.zeros((discretization, discretization))
    for i in range(discretization):
        for j in range(discretization):
            p1 = discretizedPrices1[i]
            p2 = discretizedPrices2[j]
            profitCurve[i, j] = reward_function(np.array([p1, p2]), nCustomers * np.array([conversionProbabilityP1(p1, p2), conversionProbabilityP2(p1, p2)]))
    bestPriceIndices = np.unravel_index(np.argmax(profitCurve), profitCurve.shape)
    bestPrices = [discretizedPrices1[bestPriceIndices[0]], discretizedPrices2[bestPriceIndices[1]]]
    print("Best prices: " + str(bestPrices) + "\n")
    expectedClairvoyantRewards = np.repeat(profitCurve[bestPriceIndices], T)

    X = np.arange(minPrices[0], maxPrices[0], (maxPrices[0] - minPrices[0]) / discretization)
    Y = np.arange(minPrices[1], maxPrices[1], (maxPrices[1] - minPrices[1]) / discretization)
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, profitCurve, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title("True Profit Curve")
    plt.show()

    ucbRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("GPUCB trial " + str(trial + 1))
        ucbAgent = GPUCBAgent(T, discretization, minPrices, maxPrices)
        ucbRegretPerTrial[trial, :] = testAgent(env, ucbAgent, T, expectedClairvoyantRewards)

    tsRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("GPTS trial " + str(trial + 1))
        tsAgent = GPTSAgent(T, discretization, minPrices, maxPrices)
        tsRegretPerTrial[trial, :] = testAgent(env, tsAgent, T, expectedClairvoyantRewards)

    clairvoyantRegretPerTrial = np.zeros((numTrials, T))
    for trial in range(numTrials):
        print("Best prices agent " + str(trial + 1))
        clairvoyantAgent = BestPriceAgent(bestPrices, discretizedPrices, conversionProbability)
        clairvoyantRegretPerTrial[trial, :] = testAgent(env, clairvoyantAgent, T, expectedClairvoyantRewards)

    ucbAverageRegret = ucbRegretPerTrial.mean(axis=0)
    ucbRegretStd = ucbRegretPerTrial.std(axis=0)

    tsAverageRegret = tsRegretPerTrial.mean(axis=0)
    tsRegretStd = tsRegretPerTrial.std(axis=0)

    """
    clairvoyantAverageRegret = clairvoyantRegretPerTrial.mean(axis=0)
    clairvoyantRegretStd = clairvoyantRegretPerTrial.std(axis=0)
    """

    plt.plot(np.arange(T), ucbAverageRegret, label='UCB Average Regret')
    plt.plot(np.arange(T), tsAverageRegret, label='TS Average Regret')
    """
    plt.plot(np.arange(T), clairvoyantAverageRegret, label='Best Fixed Arm Average Regret')
    """
    plt.title('cumulative regret of UCB and TS')
    plt.fill_between(np.arange(T),
                     ucbAverageRegret - ucbRegretStd / np.sqrt(numTrials),
                     ucbAverageRegret + ucbRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    plt.fill_between(np.arange(T),
                     tsAverageRegret - tsRegretStd / np.sqrt(numTrials),
                     tsAverageRegret + tsRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    """
    plt.fill_between(np.arange(T),
                     clairvoyantAverageRegret - clairvoyantRegretStd / np.sqrt(numTrials),
                     clairvoyantAverageRegret + clairvoyantRegretStd / np.sqrt(numTrials),
                     alpha=0.3)
    """
    plt.ylabel('Regret')
    plt.xlabel('$t$')
    plt.legend()
    plt.show()
