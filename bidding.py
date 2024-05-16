import numpy as np
from matplotlib import pyplot as plt


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
        self.lmbd = 1
        self.t = 0
        self.totWins = 0
        self.bidHist = np.array([])
        self.utilitityHist = np.array([])
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
        self.utilitityHist = np.append(self.utilitityHist, utility)
        self.budgetHist = np.append(self.budgetHist, self.budget)

    def returnHistory(self):
        return self.totWins, self.bidHist, self.utilitityHist, self.budgetHist

    def plotHistory(self):
        plt.plot(np.cumsum(self.utilitityHist))
        plt.xlabel('$t$')
        plt.ylabel('$Cumulative utility$')
        plt.title('Utility history')
        plt.show()

        plt.plot(self.budgetHist)
        plt.xlabel('$t$')
        plt.ylabel('$Budget$')
        plt.title('Budget history')
        plt.show()


class VCGAuction(Auction):
    def __init__(self, clickThroughRates, lambdas):
        self.clickThroughRates = clickThroughRates
        self.lambdas = lambdas
        self.nAds = len(self.clickThroughRates)
        self.nSlots = len(self.lambdas)

    def getWinners(self, bids):
        adValues = self.clickThroughRates * bids
        adRanking = np.argsort(adValues)
        winners = adRanking[-self.nSlots:]
        winnersValues = adValues[winners]
        return winners, winnersValues

    def getPaymentsPerClick(self, winners, values, bids):
        paymentsPerClick = np.zeros(self.nSlots)
        for i, w in enumerate(winners):
            Y = sum(np.delete(values, i) * self.lambdas[-self.nSlots + 1:])
            X = sum(np.delete(values * self.lambdas, i))
            paymentsPerClick[i] = (Y - X) / (self.lambdas[i] * self.clickThroughRates[w])
        return paymentsPerClick.round(2)


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


def getDeterministicClairvoyant(budget, myValuation, maxBids, nRounds):
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
