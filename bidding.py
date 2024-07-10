import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import math



class Auction:
    def __init__(self, *args, **kwargs):
        pass

    def getWinners(self, bids):
        pass

    def getPaymentsPerClick(self, winners, values, bids):
        pass

    def round(self, bids):
        winners, adValues = self.getWinners(bids)
        paymentsPerClick = self.getPaymentsPerClick(winners, adValues, bids)
        return winners, paymentsPerClick


class SecondPriceAuction(Auction):
    def __init__(self, clickThroughRates):
        self.clickThroughRates = clickThroughRates
        self.nAds = len(self.clickThroughRates)

    def getWinners(self, bids):
        adValues = self.clickThroughRates * bids
        adRanking = np.argsort(adValues)
        winner = adRanking[-1]
        return winner, adValues

    def getPaymentsPerClick(self, winners, values, bids):
        adRanking = np.argsort(values)
        second = adRanking[-2]
        payment = values[second] / self.clickThroughRates[winners]
        return payment.round(2)


class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, nRounds):
        self.valuation = valuation
        self.budget = budget
        self.eta = 1/np.sqrt(nRounds)
        self.maxRounds = nRounds
        self.rho = self.budget / self.maxRounds
        self.lmbd = 0
        self.t = 0
        self.totWins = 0
        self.bidHist = np.array([])
        self.utilityHist = np.array([])
        self.budgetHist = np.array([])

    def bid(self):
        if self.budget < 1:
            return 0
        bid = self.valuation / (self.lmbd + 1)
        self.bidHist = np.append(self.bidHist, bid)
        return bid

    def update(self, win, utility, cost):
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - cost),
                            a_min=0, a_max=1 / self.rho)
        self.budget -= cost
        self.totWins += win
        self.utilityHist = np.append(self.utilityHist, utility)
        self.budgetHist = np.append(self.budgetHist, self.budget)

    def returnHistory(self):
        return self.totWins, self.bidHist, self.utilityHist, self.budgetHist

    def plotHistory(self):
        plt.plot(np.cumsum(self.utilityHist))
        plt.xlabel('$t$')
        plt.ylabel('$Cumulative utility$')
        plt.title('Utility history')
        plt.show()

        plt.plot(self.budgetHist)
        plt.xlabel('$t$')
        plt.ylabel('$Budget$')
        plt.title('Budget history')
        plt.show()


class UCBAgent:
    def __init__(self, budget, bids, T, scaleFactor):
        self.budget = budget
        self.budgetPerRound = budget / T
        self.bids = bids
        self.bidIndices = np.arange(len(bids))
        self.maxBid = max(bids)
        self.T = T
        self.t = 1
        self.scaleFactor = scaleFactor
        self.utilityUCBs = np.repeat(sys.float_info.max, len(self.bids))
        self.costLCBs = np.repeat(0, len(bids))
        self.gamma = np.repeat(1 / len(bids), len(bids))
        self.totWins = 0
        self.bidHist = np.array([])
        self.bidIndHist = np.array([], int)
        self.utilityHist = np.array([])
        self.costHist = np.array([])
        self.budgetHist = np.array([])
        self.winningBidHist = np.array([])

    def bid(self):
        if self.budget < self.maxBid:
            return 0

        bidInd = np.random.choice(a=self.bidIndices, p=self.gamma)
        self.bidIndHist = np.append(self.bidIndHist, bidInd)
        bid = self.bids[bidInd]
        self.bidHist = np.append(self.bidHist, bid)
        return bid

    def update(self, win, utility, cost, winningBid):
        self.budget -= cost
        self.totWins += win
        self.utilityHist = np.append(self.utilityHist, utility)
        self.costHist = np.append(self.costHist, cost)
        self.budgetHist = np.append(self.budgetHist, self.budget)
        self.winningBidHist = np.append(self.winningBidHist, winningBid)

        for bidInd in self.bidIndices:
            roundsWithBid = np.where(self.bidIndHist == bidInd)[0]
            if len(roundsWithBid) > 0:
                averageBidUtility = np.mean(self.utilityHist[roundsWithBid])
                averageBidCost = np.mean(self.costHist[roundsWithBid])
                self.utilityUCBs[bidInd] = averageBidUtility + self.scaleFactor * np.sqrt(2 * np.log(self.t) / len(self.utilityHist[roundsWithBid]))
                self.costLCBs[bidInd] = np.clip(averageBidCost - self.scaleFactor * np.sqrt(2 * np.log(self.t) / len(self.costHist[roundsWithBid])), 0, np.inf)

        c = -self.utilityUCBs
        A_ub = self.costLCBs * np.ones((1, len(self.bids)))
        b_ub = self.budgetPerRound
        A_eq = np.ones((1, len(self.bids)))
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        self.gamma = res.x.T

        self.t += 1

    def returnHistory(self):
        return self.totWins, self.bidHist, self.utilityHist, self.budgetHist

    def plotHistory(self):
        plt.plot(np.cumsum(self.utilityHist))
        plt.xlabel('$t$')
        plt.ylabel('$Cumulative utility$')
        plt.title('Utility history')
        plt.show()

        plt.plot(self.budgetHist)
        plt.xlabel('$t$')
        plt.ylabel('$Budget$')
        plt.title('Budget history')
        plt.show()


def getTruthfulClairvoyant(budget, myValuation, maxBids, nRounds):
    utility = (myValuation - maxBids) * (myValuation >= maxBids)
    sortedRoundUtility = np.flip(np.argsort(utility))
    clairvoyantUtilities = np.zeros(nRounds)
    clairvoyantBidding = np.zeros(nRounds)
    clairvoyantPayments = np.zeros(nRounds)
    totPayments = 0
    roundInd = 0
    while totPayments <= budget - 1 and roundInd < nRounds:
        clairvoyantBidding[sortedRoundUtility[roundInd]] = 1
        clairvoyantUtilities[sortedRoundUtility[roundInd]] = utility[sortedRoundUtility[roundInd]]
        clairvoyantPayments[sortedRoundUtility[roundInd]] = maxBids[sortedRoundUtility[roundInd]]
        totPayments += clairvoyantPayments[sortedRoundUtility[roundInd]]
        roundInd += 1
    return clairvoyantBidding, clairvoyantUtilities, clairvoyantPayments


class FirstPriceAuction(Auction):
    def __init__(self, clickThroughRates):
        self.clickThroughRates = clickThroughRates
        self.nAds = len(self.clickThroughRates)

    def getWinners(self, bids):
        adValues = self.clickThroughRates * bids
        adRanking = np.argsort(adValues)
        winner = adRanking[-1]
        return winner, adValues

    def getPaymentsPerClick(self, winners, values, bids):
        payment = bids[winners]
        return payment.round(2)


class HedgeAgent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K
        self.a_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate * l_t)
        self.t += 1


class FFMultiplicativePacingAgent:
    def __init__(self, bids_set, valuation, budget, T, eta):
        self.bids_set = bids_set
        self.K = len(bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K) / T))
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.bids_set[self.hedge.pull_arm()]

    def update(self, f_t, c_t, m_t):
        # update hedge
        f_t_full = np.array([(self.valuation - b) * int(b >= m_t) for b in self.bids_set])
        c_t_full = np.array([b * int(b >= m_t) for b in self.bids_set])
        L = f_t_full - self.lmbd * (c_t_full - self.rho)
        range_L = 2 + (1 - self.rho) / self.rho
        self.hedge.update((2 - L) / range_L)  # hedge needs losses in [0,1]
        # update lagrangian multiplier
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0, a_max=1 / self.rho)
        # update budget
        self.budget -= c_t

def generateRandomChangingBids(T, nAdvertisers):
    advertisersBids = np.zeros((nAdvertisers, T))
    for t in range(T):
        newMaxBid = np.clip(abs(np.random.normal(0.4 * math.sin(t / 1) + 0.6, 0.1)), 0, 1)
        advertisersBids[:,t] = np.random.uniform(0,newMaxBid, nAdvertisers)
        #advertisersBids[:,t] = np.random.uniform(minBid, maxBid, nAdvertisers)
    return advertisersBids.max(axis=0), advertisersBids


def generateRandomChangingBids2(minBid, maxBid, numBids, T, numChanges, nAdvertisers):
    pattern = lambda t: 1 - np.abs(np.sin(10 * t / T))
    other_bids = np.array([np.random.uniform(0, pattern(t), size=nAdvertisers) for t in range(T)]).T
    return other_bids.max(axis=0), other_bids


def generateRandomChangingBids3(minBid, maxBid, numBids, T, numChanges, nAdvertisers):
    advertisersBids = np.zeros((nAdvertisers, T))
    for t in range(T):
        mean = np.random.uniform(0,maxBid)
        std = np.random.uniform(0, np.random.uniform(0, 1))
        advertisersBids[:, t] = np.random.normal(mean, std, nAdvertisers)
    return advertisersBids.max(axis=0), advertisersBids