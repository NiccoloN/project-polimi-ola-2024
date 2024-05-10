import numpy as np
from matplotlib import pyplot as plt


class PricingEnvironment:
    def __init__(self, conversionProbability, cost):
        self.conversionProbability = conversionProbability
        self.cost = cost

    def round(self, price_t, nCustomers_t):
        nSales_t = np.random.binomial(nCustomers_t, self.conversionProbability(price_t))
        profit_t = (price_t - self.cost) * nSales_t
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
        self.gamma = lambda t: np.log(t + 1) ** 2
        self.beta = lambda t: 1 + 0.5 * np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.N_pulls = np.zeros(discretization)
        self.t = 0

    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        ucbs = self.mu_t + self.beta(self.t) * self.sigma_t
        self.a_t = np.argmax(ucbs)
        return self.arms[self.a_t]

    def update(self, r_t, showPlot=False):
        self.N_pulls[self.a_t] += 1
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


class GPTSAgent:
    def __init__(self, T, discretization, minPrice, maxPrice):
        self.T = T
        self.arms = np.linspace(minPrice, maxPrice, discretization)
        self.gp = RBFGaussianProcess(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros(discretization)
        self.sigma_t = np.zeros(discretization)
        self.N_pulls = np.zeros(discretization)
        self.t = 0

    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        samples = np.random.normal(self.mu_t, self.sigma_t)
        self.a_t = np.argmax(samples)
        return self.arms[self.a_t]

    def update(self, r_t, showPlot=False):
        self.N_pulls[self.a_t] += 1
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


class ClairvoyantAgent:
    def __init__(self, action):
        self.action = action

    def pull_arm(self):
        return self.action

    def update(self, r_t, showPlot=False):
        return
