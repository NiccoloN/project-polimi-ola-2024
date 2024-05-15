import numpy as np


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


def getClairvoyantTruthful(budget, myValuation, maxBids_t, nUsers):
    utility = (myValuation - maxBids_t) * (myValuation >= maxBids_t)
    sortedRoundUtility = np.flip(np.argsort(utility))
    clairvoyantUtilities = np.zeros(nUsers)
    clairvoyantBids = np.zeros(nUsers)
    clairvoyantPayments = np.zeros(nUsers)
    c = 0
    i = 0
    while c <= budget - 1 and i < nUsers:
        clairvoyantBids[sortedRoundUtility[i]] = 1
        clairvoyantUtilities[sortedRoundUtility[i]] = utility[sortedRoundUtility[i]]
        clairvoyantPayments[sortedRoundUtility[i]] = maxBids_t[sortedRoundUtility[i]]
        c += maxBids_t[sortedRoundUtility[i]]
        i += 1
    return clairvoyantBids, clairvoyantUtilities, clairvoyantPayments
