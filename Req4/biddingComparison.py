from bidding import *
import random
import numpy as np
import matplotlib.pyplot as plt


def biddingComparison(nSlots):
    eta = 1 / np.sqrt(nUsers)
    gmpAgent = FFMultiplicativePacingAgent(possibleBids, myValuation, B, T, eta, True)
    mpAgent = MultiplicativePacingAgent(myValuation, B, T, True)
    ucbAgent = UCBAgent(B, possibleBids, T, 0.016, True)

    otherAgents = []
    for i in range(nAdvertisers - 3):
        if i % 3 == 0:
            otherAgents.append(FFMultiplicativePacingAgent(possibleBids, myValuation, B, T, eta, True))
        elif i % 3 == 1:
            otherAgents.append(MultiplicativePacingAgent(myValuation, B, T, True))
        else:
            otherAgents.append(UCBAgent(B, possibleBids, T, 0.016, True))

    auction = GeneralizedFirstPriceAuction(nSlots, np.ones(nAdvertisers))

    gmpUtilities = np.array([])
    gmpBids = np.array([])
    gmpPayments = np.array([])
    gmpTotalWins = 0

    mpUtilities = np.array([])
    mpBids = np.array([])
    mpPayments = np.array([])
    mpTotalWins = 0

    ucbUtilities = np.array([])
    ucbBids = np.array([])
    ucbPayments = np.array([])
    ucbTotalWins = 0

    bids = np.zeros((nAdvertisers, T))
    m_t = np.zeros(T)

    for u in range(nUsers):
        gmpBid = gmpAgent.bid()
        mpBid = mpAgent.bid()
        ucbBid = ucbAgent.bid()

        bids[0:3, u] = np.array([gmpBid, mpBid, ucbBid])
        bids[3:nAdvertisers, u] = list(map(lambda agent: agent.bid(), otherAgents))
        m_t[u] = max(bids[:, u])

        winners, paymentsPerClick = auction.round(bids=bids[:, u])

        def getWinUtilityAndCost(agentId):
            win = agentId in winners
            bid = bids[agentId, u]
            return win, (myValuation - bid) * win, bid * win

        gmpWin, gmpf_t, gmpc_t = getWinUtilityAndCost(0)
        mpWin, mpf_t, mpc_t = getWinUtilityAndCost(1)
        ucbWin, ucbf_t, ucbc_t = getWinUtilityAndCost(2)

        gmpAgent.update(gmpWin, gmpf_t, gmpc_t, m_t[u])
        mpAgent.update(mpWin, mpf_t, mpc_t)
        ucbAgent.update(ucbWin, ucbf_t, ucbc_t)

        for agentId in range(3, nAdvertisers):
            otherAgents[agentId-3].update(*getWinUtilityAndCost(agentId), m_t[u])

        gmpUtilities = np.append(gmpUtilities, gmpf_t)
        gmpBids = np.append(gmpBids, gmpBid)
        gmpPayments = np.append(gmpPayments, gmpc_t)
        gmpTotalWins += gmpWin

        mpUtilities = np.append(mpUtilities, mpf_t)
        mpBids = np.append(mpBids, mpBid)
        mpPayments = np.append(mpPayments, mpc_t)
        mpTotalWins += mpWin

        ucbUtilities = np.append(ucbUtilities, ucbf_t)
        ucbBids = np.append(ucbBids, ucbBid)
        ucbPayments = np.append(ucbPayments, ucbc_t)
        ucbTotalWins += ucbWin

    gmpCumulativePayments = np.cumsum(gmpPayments)
    mpCumulativePayments = np.cumsum(mpPayments)
    ucbCumulativePayments = np.cumsum(ucbPayments)

    expectedClairvoyantUtilities, expectedClairvoyantBids = getAdversarialNonTruthfulClairvoyant(T, possibleBids, m_t, myValuation, rho)

    gmpCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - gmpUtilities)
    mpCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - mpUtilities)
    ucbCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - ucbUtilities)
    plt.plot(mpCumulativeRegret, label="MP Cumulative Regret")
    plt.plot(gmpCumulativeRegret, label="GMP Cumulative Regret")
    plt.plot(ucbCumulativeRegret, label="UCB Cumulative Regret")
    plt.xlabel('$t$')
    plt.ylabel('$\sum R_t$')
    plt.legend()
    plt.title('Cumulative Regret (' + str(nSlots) + ' slots)')
    plt.show()

    plt.plot(mpCumulativePayments, label="MP Cumulative Payments")
    plt.plot(gmpCumulativePayments, label="GMP Cumulative Payments")
    plt.plot(ucbCumulativePayments, label="UCB Cumulative Payments")
    plt.xlabel('$t$')
    plt.ylabel('$\sum c_t$')
    plt.axhline(B, color='red', label='Budget')
    plt.legend()
    plt.title('Cumulative Payments (' + str(nSlots) + ' slots)')
    plt.show()

    plt.plot(mpBids, 'o', markersize=1, label="MP Agent Bids")
    plt.plot(gmpBids, 'o', markersize=1, label="GMP Agent Bids")
    plt.plot(ucbBids, 'o', markersize=1, label="UCB Agent Bids")
    plt.plot(expectedClairvoyantBids, markersize=1, label="Clairvoyant Bids")
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Bids$')
    plt.ylim(0, 1)
    plt.title('Bids History (' + str(nSlots) + ' slots)')
    plt.show()

    plt.plot(m_t, 'o', markersize=1, label="Max Bids")
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Bids$')
    plt.ylim(0, 1)
    plt.title('Max Bids (' + str(nSlots) + ' slots)')
    plt.show()


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)

    nAdvertisers = 9
    minBid = 0.2
    maxBid = 1
    myValuation = 0.8
    possibleBids = np.arange(0, 1, 0.01)
    numBids = len(possibleBids)
    nUsers = 1000
    B = nUsers/8
    T = nUsers
    rho = B/T

    slots = [1, 2, 3]

    for nSlots in slots:
        biddingComparison(nSlots)
