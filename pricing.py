import math
import random
import numpy as np
from matplotlib import pyplot as plt


class StochasticEnvironment:
    def __init__(self, conversionProbability, cost):
        self.conversionProbability = conversionProbability
        self.cost = cost

    def round(self, price_t, nCustomers_t):
        nSales_t = np.random.binomial(nCustomers_t, self.conversionProbability(price_t))
        profit_t = (price_t - self.cost) * nSales_t
        return nSales_t, profit_t


def generateRandomDemandCurve(minPrice, maxPrice, numPrices):
    curve = np.zeros(numPrices)
    start = 0
    factor1 = random.choice([15, 20, 25, 30])
    factor2 = random.choice([5, 10, 15])
    for i in range(0, numPrices):
        start += math.log(random.random()) / factor1 / (numPrices - i) * factor2
        curve[i] = math.exp(start) * (maxPrice - minPrice) + minPrice

    curve = curve - np.min(curve)
    curve = curve / np.max(curve)
    return curve


def generateRandomChangingDemandCurve(minPrice, maxPrice, numPrices, T, numChanges, plot):
    changePoints = np.random.normal(T / numChanges, T / numChanges / 7, size=numChanges).astype(int)
    changePoints = changePoints * np.arange(numChanges)
    changePoints = np.round(T * (changePoints - abs(min(changePoints))) / (max(changePoints) - abs(min(changePoints)))).astype(int)
    sortedChangePoints = np.sort(changePoints)

    mu = np.zeros((T, numPrices))
    demandCurve = generateRandomDemandCurve(minPrice, maxPrice, numPrices)

    if plot:
        plt.plot(demandCurve)
        plt.ylim((0, maxPrice))
        plt.show()

    changePointIndex = 0
    changePoint = 0
    for i in range(T):
        if i >= changePoint:
            demandCurve = generateRandomDemandCurve(minPrice, maxPrice, numPrices)
            if changePoint < sortedChangePoints[-1]:
                changePointIndex += 1
                changePoint = sortedChangePoints[changePointIndex]
            else:
                changePoint = T+1

            if plot:
                plt.plot(demandCurve)
                plt.ylim((0, maxPrice))
                plt.show()
        mu[i, :] = demandCurve
    return mu, sortedChangePoints


class NonStationaryBernoulliEnvironment:
    def __init__(self, cost, minPrice, maxPrice, prices, numChanges, T, seed, plot):
        np.random.seed(seed)
        self.cost = cost
        self.pricesArray = prices
        self.K = prices.size
        self.mu, self.sortedChangePoints = generateRandomChangingDemandCurve(minPrice, maxPrice, self.K, T, numChanges, plot)
        self.t = 0

    def round(self, price_t, nCustomers_t):
        nSales_t = np.random.binomial(n=nCustomers_t, p=self.mu[self.t, np.where(self.pricesArray == price_t)])
        profit_t = (price_t - self.cost) * nSales_t
        self.t += 1
        return nSales_t, profit_t


class RBFGaussianProcess:
    def __init__(self, scale=1, reg=1e-2):
        self.scale = scale
        self.reg = reg
        self.k_xx_inv = None

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
        a_ = a.reshape(-1, 1)
        b_ = b.reshape(-1, 1)
        output = -1 * np.ones((a_.shape[0], b_.shape[0]))
        for i in range(a_.shape[0]):
            output[i, :] = np.power(a_[i] - b_, 2).ravel()
        return np.exp(-self.scale * output)

    def fit(self, x=np.array([]), y=np.array([])):
        x, y = np.array(x), np.array(y)
        if self.k_xx_inv is None:
            self.y = y.reshape(-1, 1)
            self.x = x.reshape(-1, 1)
            k_xx = self.rbf_kernel(self.x, self.x) + self.reg * np.eye(self.x.shape[0])
            self.k_xx_inv = np.linalg.inv(k_xx)
        else:
            B = self.rbf_kernel(self.x, x)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
            self.k_xx_inv = self.rbf_kernel_incr_inv(B, B.T, np.array([1 + self.reg]))

        return self

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()


class GPUCBAgent:
    def __init__(self, T, discretization, minPrice, maxPrice):
        self.T = T
        self.arms = np.linspace(minPrice, maxPrice, discretization)
        self.gp = RBFGaussianProcess(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros(discretization)
        self.sigma_t = np.zeros(discretization)
        self.ucbs_t = np.zeros(discretization)
        self.gamma = lambda t: np.log(t + 1) ** 2
        self.beta = lambda t: 1 + 0.5 * np.sqrt(2 * (self.gamma(t) + 1 + np.log(t)))
        self.nPulls = np.zeros(discretization)
        self.t = 1

    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        self.ucbs_t = self.mu_t + self.beta(self.t) * self.sigma_t
        self.a_t = np.argmax(self.ucbs_t)
        return self.arms[self.a_t]

    def update(self, r_t, showPlot=False):
        self.nPulls[self.a_t] += 1
        self.action_hist = np.append(self.action_hist, self.arms[self.a_t])
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp = self.gp.fit(self.arms[self.a_t], r_t)
        self.t += 1

        if showPlot:
            mu, sigma = self.gp.predict(self.arms)

            plt.plot(self.arms, mu)
            plt.fill_between(self.arms, mu - sigma, mu + sigma, alpha=0.3)
            plt.suptitle(f'Estimated Profit - {self.t + 1} samples')
            plt.scatter(self.action_hist, self.reward_hist)
            plt.show()

    def getEstimatedRewardMean(self):
        return self.ucbs_t


class GPTSAgent:
    def __init__(self, T, discretization, minPrice, maxPrice):
        self.T = T
        self.arms = np.linspace(minPrice, maxPrice, discretization)
        self.gp = RBFGaussianProcess(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros(discretization)
        self.samples_t = np.zeros(discretization)
        self.sigma_t = np.zeros(discretization)
        self.nPulls = np.zeros(discretization)
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
        self.gp = self.gp.fit(self.arms[self.a_t], r_t)
        self.t += 1

        if showPlot:
            mu, sigma = self.gp.predict(self.arms)

            plt.plot(self.arms, mu)
            plt.fill_between(self.arms, mu - sigma, mu + sigma, alpha=0.3)
            plt.suptitle(f'Estimated Profit - {self.t + 1} samples')
            plt.scatter(self.action_hist, self.reward_hist)
            plt.show()

    def getEstimatedRewardMean(self):
        return self.samples_t


class ClairvoyantAgent:
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


class SWUCBAgent:
    def __init__(self, discretizedPrices, T, W, range=1):
        self.K = discretizedPrices.size
        self.discretizedPrices = discretizedPrices
        self.T = T
        self.W = W
        self.range = range
        self.a_t = None
        self.armsHistory = np.zeros(T)
        self.rewardsCache = np.repeat(np.nan, repeats=self.K * W).reshape(W, self.K)
        self.N_pulls = np.zeros(self.K)
        self.t = 0

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            nPulls_w = self.W - np.isnan(self.rewardsCache).sum(axis=0)
            avgRewards_w = np.nanmean(self.rewardsCache, axis=0)
            ucbs = avgRewards_w + self.range * np.sqrt(
                2 * np.log(self.W) / nPulls_w)  # there's a typo in the slides, log(T) -> log(W)
            self.a_t = np.argmax(ucbs)
        self.armsHistory[self.t] = self.a_t
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
        self.resetTimes = np.array([0])
        self.N_pulls = np.zeros(self.K)
        self.all_rewards = [[] for _ in np.arange(self.K)]
        self.counters = np.repeat(M, self.K)
        self.average_rewards = np.zeros(self.K)
        self.n_resets = 0
        self.n_t = 0    # Total number of pulls
        self.t = 0


    def pull_arm(self):
        if (self.counters > 0).any():
            for a in np.arange(self.K):
                if self.counters[a] > 0:
                    self.counters[a] -= 1
                    self.a_t = a
                    break
        else:
            if np.random.random() <= 1 - self.alpha:
                ucbs = self.average_rewards + self.range * np.sqrt(np.log(self.n_t) / self.N_pulls)
                self.a_t = np.argmax(ucbs)
            else:
                self.a_t = np.random.choice(np.arange(self.K))    # Extra exploration
        return self.discretizedPrices[self.a_t]

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.all_rewards[self.a_t].append(r_t)
        if self.counters[self.a_t] == 0:
            if self.change_detection():
                self.n_resets += 1
                self.N_pulls[self.a_t] = 0
                self.average_rewards[self.a_t] = 0
                self.counters[self.a_t] = self.M
                self.all_rewards = [[] for _ in np.arange(self.K)]
                self.lastResetTime = self.t
                self.resetTimes = np.append(self.resetTimes, self.t)

            else:
                self.average_rewards[self.a_t] += (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.n_t = sum(self.N_pulls)
        self.t += 1

    def change_detection(self):
        # CUSUM CD sub-routine. This function returns 1 if there's evidence that the last pulled arm has its average reward changed
        u_0 = np.mean(self.all_rewards[self.a_t][:self.M])
        sp, sm = (np.array(self.all_rewards[self.a_t][self.M:]) - u_0, u_0 - np.array(self.all_rewards[self.a_t][self.M:]))
        gp, gm = 0, 0
        for sp_, sm_ in zip(sp, sm):
            gp, gm = max([0, gp + sp_]), max([0, gm + sm_])
            if max([gp, gm]) >= self.h:
                return True
        return False
