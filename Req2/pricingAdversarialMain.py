from bidding import generateRandomChangingBids
from pricing import *
import matplotlib.cm as cm


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
    cost = 0.2
    minPrice = cost
    maxPrice = 1
    nCustomers = 100
    priceRange = maxPrice - minPrice
    discretization = 1000
    discretizedPrices = np.linspace(minPrice, maxPrice, discretization)
    numDemandChanges = 9

    T = 1000
    numTrials = 10
    env = NonStationaryBernoulliEnvironment(minPrice, maxPrice, discretization, numDemandChanges, T, 1, True)

    # Best policy in Hindsight:
    best_rewards = np.array(env.mu).max(axis=0)  # we take the max over every single round
    best_cum_rew = sum(best_rewards)
    best_policy = np.array(env.mu).argmax(axis=0)
    print(f'Best possible cumulative reward: {best_cum_rew}')  # is higher than the cumulative reward of the best arm in hindsight

    t = np.arange(T)
    for i in range(discretization):
        plt.plot(t, env.mu[i], label=f'$\mu_{i}$')
    plt.plot(t, best_rewards, label=f'$\mu^*$')

    plt.legend()
    plt.xlabel('$t$')
    plt.show()

    m_t, changingPoints, check = generateRandomChangingBids(0.2, 1, 1000, T, 9)
    m_t = (m_t + abs(min(m_t))) / (max(m_t) + abs(min(m_t)))
    t = np.arange(T)
    print(m_t)
    print(check)

    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(8):
        xRange = np.arange(changingPoints[i-1], changingPoints[i])
        plt.plot(xRange, m_t[changingPoints[i-1]:changingPoints[i]], 'o', color=colors[i])


    plt.legend()
    plt.xlabel('$t$')
    plt.show()
