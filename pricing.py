import math
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import fsolve


class StochasticEnvironment:
    def __init__(self, conversionProbabilityFunctions, costs, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.conversionProbabilityFunctions = conversionProbabilityFunctions
        self.costs = costs

    def round(self, prices_t, nCustomers_t):
        nSales_t = np.repeat(0, len(self.conversionProbabilityFunctions))
        profit_t = 0
        for i, f in enumerate(self.conversionProbabilityFunctions):
            price_t = prices_t[i]
            cost = self.costs[i]
            nSales_t[i] += self.rng.binomial(nCustomers_t, f(*prices_t))
            profit_t += (price_t - cost) * nSales_t[i]
        return nSales_t, profit_t

    def reset(self):
        self.rng.seed(self.seed)


def curveWithPeakEquation(x, param1, param2, param3):
    return (param1 * np.log(x)) ** 2 - param2 * np.sin(x) - x ** 2 + param3 * x


def generateRandomCurveWithPeak(minPrice, maxPrice, numPrices, rng):
    curve = np.zeros(numPrices)

    param1 = rng.uniform(0.35, 0.65)
    param2 = rng.uniform(0.1, 1.0)
    param3 = rng.uniform(2.0, 4.0)

    initialGuess = 3
    finalValue = fsolve(curveWithPeakEquation, initialGuess, (param1, param2, param3))
    initialValue = 0.02

    for ind, x in enumerate(np.linspace(initialValue, finalValue, num=numPrices)):
        curve[ind] = curveWithPeakEquation(x, param1, param2, param3)

    curve = curve - np.min(curve)
    curve = curve / np.max(curve)
    return curve


def convexCurveEquation(x, param1, param2):
    return param1 * 2.71 ** (-x ** param2)


def generateRandomConvexCurve(minPrice, maxPrice, numPrices, rng):
    curve = np.zeros(numPrices)

    param1 = rng.uniform(0.5, 3.0)
    param2 = rng.uniform(0.8, 1.6)

    initialValue = 0.02
    finalValue = 2.0

    for ind, x in enumerate(np.linspace(initialValue, finalValue, num=numPrices)):
        curve[ind] = convexCurveEquation(x, param1, param2)

    curve = curve - np.min(curve)
    curve = curve / np.max(curve)
    return curve


def generateRandomDescendingCurve(minPrice, maxPrice, numPrices, rng):
    curve = np.zeros(numPrices)
    start = 0
    factor1 = rng.choice([15, 20, 25, 30])
    factor2 = rng.choice([5, 10, 15])
    for i in range(0, numPrices):
        start += math.log(rng.random()) / factor1 / (numPrices - i) * factor2
        curve[i] = math.exp(start) * (maxPrice - minPrice) + minPrice

    curve = curve - np.min(curve)
    curve = curve / np.max(curve)
    return curve


def generateRandomChangingDemandCurve(minPrice, maxPrice, numPrices, T, numChanges, rng, plot):
    changePoints = rng.normal(T / numChanges, T / numChanges / 7, size=numChanges).astype(int)
    changePoints = changePoints * np.arange(numChanges)
    changePoints = np.round(
        T * (changePoints - abs(min(changePoints))) / (max(changePoints) - abs(min(changePoints)))).astype(int)
    sortedChangePoints = np.sort(changePoints)

    mu = np.zeros((T, numPrices))

    convexCurveProbability = 0.4
    curveWithPeakProbability = 0.3

    randomValue = rng.random()
    if randomValue <= curveWithPeakProbability:
        demandCurve = generateRandomCurveWithPeak(minPrice, maxPrice, numPrices, rng)
    elif randomValue <= convexCurveProbability + curveWithPeakProbability:
        demandCurve = generateRandomConvexCurve(minPrice, maxPrice, numPrices, rng)
    else:
        demandCurve = generateRandomDescendingCurve(minPrice, maxPrice, numPrices, rng)

    if plot:
        plt.plot(demandCurve)
        plt.ylim((0, maxPrice))
        plt.show()

    changePointIndex = 0
    changePoint = 0
    for i in range(T):
        if i >= changePoint:
            randomValue = rng.random()
            if randomValue <= curveWithPeakProbability:
                demandCurve = generateRandomCurveWithPeak(minPrice, maxPrice, numPrices, rng)
            elif randomValue <= convexCurveProbability + curveWithPeakProbability:
                demandCurve = generateRandomConvexCurve(minPrice, maxPrice, numPrices, rng)
            else:
                demandCurve = generateRandomDescendingCurve(minPrice, maxPrice, numPrices, rng)

            if changePoint < sortedChangePoints[-1]:
                changePointIndex += 1
                changePoint = sortedChangePoints[changePointIndex]
            else:
                changePoint = T + 1

            if plot:
                plt.plot(demandCurve)
                plt.ylim((0, maxPrice))
                plt.title("Change at time t = " + str(i))
                plt.show()
        mu[i, :] = demandCurve
    return mu, sortedChangePoints


class NonStationaryBernoulliEnvironment:
    def __init__(self, cost, minPrice, maxPrice, prices, numChanges, T, seed, plot):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.cost = cost
        self.pricesArray = prices
        self.K = prices.size
        self.mu, self.sortedChangePoints = generateRandomChangingDemandCurve(minPrice, maxPrice, self.K, T, numChanges,
                                                                             self.rng, plot)
        self.reset()

    def round(self, price_t, nCustomers_t):
        nSales_t = self.rng.binomial(n=nCustomers_t, p=self.mu[self.t, np.where(self.pricesArray == price_t)])
        profit_t = (price_t - self.cost) * nSales_t
        self.t += 1
        return nSales_t, profit_t

    def reset(self):
        self.rng.seed(self.seed)
        self.t = 0


class RBFGaussianProcess:
    def __init__(self, nFeatures, scale=1, reg=1e-2):
        self.nFeatures = nFeatures
        self.scale = scale
        self.reg = reg
        self.k_xx_inv = np.array([]).reshape(0, 0)
        self.x = np.array([]).reshape(0, nFeatures)
        self.y = np.array([]).reshape(0, 1)
        self.start = 1

    def rbf_kernel_incr_inv(self, B, C, D):
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = - self.k_xx_inv @ B @ temp
        block3 = - temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        res = np.concatenate((res1, res2), axis=0)
        return res

    def rbf_kernel(self, a, b):
        output = -1 * np.ones((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            output[i, :] = np.power(np.linalg.norm(a[i] - b, axis=1), 2)
        return np.exp(-self.scale * output)

    def fit(self, x=np.array([]), y=np.array([])):
        x, y = np.array(x), np.array(y)

        B = self.rbf_kernel(self.x, x)
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))
        self.k_xx_inv = self.rbf_kernel_incr_inv(B, B.T, np.array([1 + self.reg]))

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()


class GPUCBAgent:
    def __init__(self, T, discretization, minPrices, maxPrices):
        self.T = T
        self.nFeatures = len(minPrices)
        self.discretization = discretization
        self.minPrices = minPrices
        self.maxPrices = maxPrices

        self.discretizedPrices = [np.linspace(minPrices[i], maxPrices[i], discretization) for i in range(self.nFeatures)]
        self.arms = self.discretizedPrices[0]
        for i in range(1, self.nFeatures):
            self.arms = np.array(
                [list(item) for item in list(itertools.product(self.arms, self.discretizedPrices[i]))])

        self.gp = RBFGaussianProcess(self.nFeatures, scale=2)
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros((discretization, discretization))
        self.sigma_t = np.zeros((discretization, discretization))
        self.ucbs_t = np.zeros((discretization, discretization))
        self.gamma = lambda t: np.log(t + 1) ** 2
        self.beta = lambda t: 1 + 0.5 * np.sqrt(2 * (self.gamma(t) + 1 + np.log(t)))
        self.nPulls = np.zeros(self.arms.shape)
        self.t = 1

    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        self.ucbs_t = self.mu_t + self.beta(self.t) * self.sigma_t
        self.a_t = np.argmax(self.ucbs_t)

        """
        ucbs_t = self.ucbs_t.reshape(self.discretization, self.discretization)

        X = np.arange(self.minPrices[0], self.maxPrices[0],
                      (self.maxPrices[0] - self.minPrices[0]) / self.discretization)
        Y = np.arange(self.minPrices[1], self.maxPrices[1],
                      (self.maxPrices[1] - self.minPrices[1]) / self.discretization)
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, ucbs_t, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()
        """

        return self.arms[self.a_t]

    def update(self, r_t, showPlot=False):
        self.nPulls[self.a_t] += 1
        self.action_hist = np.append(self.action_hist, self.arms[self.a_t])
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp.fit(self.arms[self.a_t].reshape(1, self.nFeatures), r_t)
        self.t += 1

        if showPlot:
            mu, sigma = self.gp.predict(self.arms)

            if self.nFeatures == 1:
                plt.plot(mu)
                plt.fill_between(mu - sigma, mu + sigma, alpha=0.3)
                plt.suptitle(f'Estimated Profit - {self.t + 1} samples')
                plt.scatter(self.action_hist, self.reward_hist)
                plt.show()
            elif self.nFeatures == 2:
                mu = mu.reshape(self.discretization, self.discretization)

                X = np.arange(self.minPrices[0], self.maxPrices[0], (self.maxPrices[0] - self.minPrices[0]) / self.discretization)
                Y = np.arange(self.minPrices[1], self.maxPrices[1], (self.maxPrices[1] - self.minPrices[1]) / self.discretization)
                X, Y = np.meshgrid(X, Y)

                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.plot_surface(X, Y, mu, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                plt.show()
                # plt.savefig("curve_" + str(self.t-1) + ".png")

    def getEstimatedRewardMean(self):
        return self.ucbs_t


class GPTSAgent:
    def __init__(self, T, discretization, minPrices, maxPrices):
        self.T = T
        self.nFeatures = len(minPrices)
        self.discretization = discretization
        self.minPrices = minPrices
        self.maxPrices = maxPrices

        self.discretizedPrices = [np.linspace(minPrices[i], maxPrices[i], discretization) for i in range(self.nFeatures)]
        self.arms = self.discretizedPrices[0]
        for i in range(1, self.nFeatures):
            self.arms = np.array(
                [list(item) for item in list(itertools.product(self.arms, self.discretizedPrices[i]))])

        self.gp = RBFGaussianProcess(self.nFeatures, scale=2)
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros((discretization, discretization))
        self.sigma_t = np.zeros((discretization, discretization))
        self.samples_t = np.zeros((discretization, discretization))
        self.nPulls = np.zeros(self.arms.shape)
        self.t = 0

    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        self.samples_t = np.random.normal(self.mu_t, self.sigma_t)
        self.a_t = np.argmax(self.samples_t)
        return self.arms[self.a_t]

    def update(self, r_t, showPlot=False):
        self.nPulls[self.a_t] += 1
        self.action_hist = np.append(self.action_hist, self.arms[self.a_t])
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp.fit(self.arms[self.a_t].reshape(1, self.nFeatures), r_t)
        self.t += 1

        if showPlot:
            mu, sigma = self.gp.predict(self.arms)

            if self.nFeatures == 1:
                plt.plot(mu)
                plt.fill_between(mu - sigma, mu + sigma, alpha=0.3)
                plt.suptitle(f'Estimated Profit - {self.t + 1} samples')
                plt.scatter(self.action_hist, self.reward_hist)
                plt.show()
            elif self.nFeatures == 2:
                mu = mu.reshape(self.discretization, self.discretization)

                X = np.arange(self.minPrices[0], self.maxPrices[0],
                              (self.maxPrices[0] - self.minPrices[0]) / self.discretization)
                Y = np.arange(self.minPrices[1], self.maxPrices[1],
                              (self.maxPrices[1] - self.minPrices[1]) / self.discretization)
                X, Y = np.meshgrid(X, Y)

                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.plot_surface(X, Y, mu, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                plt.show()
                plt.close()

    def getEstimatedRewardMean(self):
        return self.samples_t


class BestPriceAgent:
    def __init__(self, action, discretizedPrices, conversionProbability):
        self.action = action
        self.discretizedPrices = discretizedPrices
        self.conversionProbability = conversionProbability

    def pull_arm(self):
        return self.action

    def update(self, r_t, showPlot=False):
        return

    def getEstimatedRewardMean(self):
        return self.conversionProbability(self.discretizedPrices)


class EXP3Agent:
    def __init__(self, discretizedPrices, T, learning_rate):
        self.discretizedPrices = discretizedPrices
        self.K = len(discretizedPrices)
        self.learning_rate = learning_rate
        self.weights = np.ones(self.K)
        self.a_t = None
        self.x_t = np.ones(self.K) / self.K
        self.nPulls = np.zeros(self.K)
        self.pricesHistory = np.zeros(T)
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        self.pricesHistory[self.t] = self.discretizedPrices[self.a_t]
        return self.discretizedPrices[self.a_t]

    def update(self, r_t):
        l_t = 1 - r_t
        l_t_tilde = l_t / self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp(-self.learning_rate * l_t_tilde)
        self.nPulls[self.a_t] += 1
        self.t += 1

    def getEstimatedRewardMean(self):
        return self.x_t


def getAdversarialClairvoyant(discretizedPrices, T, env, nCustomersArray):
    rewards = np.zeros((len(discretizedPrices), T))

    for i, price in enumerate(discretizedPrices):
        env.reset()
        for t in range(T):
            demand_t, rewards[i, t] = env.round(price, nCustomersArray[t])

    env.reset()
    bestPriceIndex = np.argmax(rewards.sum(axis=1))
    bestPriceRewards = rewards[bestPriceIndex, :]
    nonZeroRewardIndices = np.where(bestPriceRewards != 0)
    bestPriceRewards[nonZeroRewardIndices] = bestPriceRewards[nonZeroRewardIndices] / nCustomersArray[nonZeroRewardIndices]
    return discretizedPrices[bestPriceIndex], bestPriceRewards


class SWUCBAgent:
    def __init__(self, discretizedPrices, T, W, range=1):
        self.K = discretizedPrices.size
        self.discretizedPrices = discretizedPrices
        self.T = T
        self.W = W
        self.range = range
        self.a_t = None
        self.pricesHistory = np.zeros(T)
        self.rewardsCache = np.repeat(np.nan, repeats=self.K * W).reshape(W, self.K)
        self.N_pulls = np.zeros(self.K)
        self.t = 0
        self.ucbsHistory = np.zeros((self.T, self.K))

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            nPulls_w = self.W - np.isnan(self.rewardsCache).sum(axis=0)
            avgRewards_w = np.nanmean(self.rewardsCache, axis=0)
            ucbs = avgRewards_w + self.range * np.sqrt(2 * np.log(self.W) / nPulls_w)
            self.a_t = np.argmax(ucbs)
            self.ucbsHistory[self.t, :] = ucbs
        self.pricesHistory[self.t] = self.discretizedPrices[self.a_t]
        return self.discretizedPrices[self.a_t]

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.rewardsCache = np.delete(self.rewardsCache, (0), axis=0)  # remove oldest observation
        new_samples = np.repeat(np.nan, self.K)
        new_samples[self.a_t] = r_t
        self.rewardsCache = np.vstack((self.rewardsCache, new_samples))  # add new observation
        self.t += 1


class CUSUMUCBAgent:
    def __init__(self, discretizedPrices, T, M, h, alpha=0.1, range=1):
        self.discretizedPrices = discretizedPrices
        self.K = discretizedPrices.size
        self.T = T
        self.M = M
        self.h = h
        self.alpha = alpha
        self.range = range
        self.a_t = None
        self.lastResetTime = 0
        self.resetTimes = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.all_rewards = [[] for _ in np.arange(self.K)]
        self.counters = np.repeat(M, self.K)
        self.average_rewards = np.zeros(self.K)
        self.n_resets = np.zeros(self.K)
        self.n_t = 0  # Total number of pulls
        self.t = 0
        self.resetTimesHistory = np.array([])
        self.pricesHistory = np.array([])
        self.ucbsHistory = np.zeros((self.T, self.K))
        self.ucbs = np.full(self.K, np.inf)

    def pull_arm(self):
        if (self.counters > 0).any():
            for a in np.arange(self.K):
                if self.counters[a] > 0:
                    self.counters[a] -= 1
                    if self.t <= self.M * self.K:
                        self.a_t = self.t % self.K
                    else:
                        self.a_t = a
                    break
        else:
            if np.random.random() <= 1 - self.alpha:
                self.ucbs = self.average_rewards + self.range * np.sqrt(np.log(self.n_t) / self.N_pulls)
                self.a_t = np.argmax(self.ucbs)
            else:
                self.a_t = np.random.choice(np.arange(self.K))  # Extra exploration

        self.ucbsHistory[self.t, :] = self.ucbs
        self.pricesHistory = np.append(self.pricesHistory, self.discretizedPrices[self.a_t])
        return self.discretizedPrices[self.a_t]

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.all_rewards[self.a_t].append(r_t)
        if self.counters[self.a_t] == 0:
            if self.change_detection():
                self.n_resets[self.a_t] += 1
                self.N_pulls[self.a_t] = 0
                self.average_rewards[self.a_t] = 0
                self.counters[self.a_t] = self.M
                self.all_rewards[self.a_t] = []
                self.lastResetTime = self.t
                self.resetTimes[self.a_t] = self.t
                self.resetTimesHistory = np.append(self.resetTimesHistory, self.t)
                self.ucbs[self.a_t] = np.inf

            else:
                self.average_rewards[self.a_t] += (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.n_t = sum(self.N_pulls)
        self.t += 1

    def change_detection(self):
        # CUSUM CD sub-routine. This function returns 1 if there's evidence that the last pulled arm has its average reward changed
        u_0 = np.mean(self.all_rewards[self.a_t][:self.M])
        sp, sm = (
        np.array(self.all_rewards[self.a_t][self.M:]) - u_0, u_0 - np.array(self.all_rewards[self.a_t][self.M:]))
        gp, gm = 0, 0
        for sp_, sm_ in zip(sp, sm):
            gp, gm = max([0, gp + sp_]), max([0, gm + sm_])
            if max([gp, gm]) >= self.h:
                return True
        return False
