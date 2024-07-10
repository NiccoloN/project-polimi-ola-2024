from bidding import *
from pricing import *
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

if __name__ == '__main__':
    nAdvertisers = 3
    minBid = 0.2
    maxBid = 1
    myValuation = 0.8
    othersValuation = 0.6
    possibleBids = np.arange(0, 1, 0.01)
    numBids = len(possibleBids)
    nUsers = 1000
    B = nUsers/5
    T = nUsers
    rho = B/T

    eta = 1 / np.sqrt(nUsers)
    gmpAgent = FFMultiplicativePacingAgent(bids_set=possibleBids, valuation=myValuation, budget=B, T=nUsers, eta=eta)
    mpAgent = MultiplicativePacingAgent(myValuation, B, T)
    ucbAgent = UCBAgent(B, possibleBids, T, 0.016)

    auction = FirstPriceAuction(np.ones(nAdvertisers))

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
    winHistory = np.zeros(T)
    np.random.seed(T)

    for u in range(nUsers):
        gmpBid = gmpAgent.bid()
        mpBid = gmpAgent.bid()
        ucbBid = gmpAgent.bid()

        bids[:,u] = np.array([gmpBid, mpBid, ucbBid])
        m_t[u] = max(bids[:,u])
        winners, paymentsPerClick = auction.round(bids=bids[:,u])
        gmpWin = int(winners == 0)
        mpWin = int(winners == 1)
        ucbWin = int(winners == 2)
        winHistory[u] = winners

        gmpf_t, gmpc_t = (myValuation - gmpBid) * gmpWin, gmpBid * gmpWin
        mpf_t, mpc_t = (myValuation - mpBid) * mpWin, mpBid * mpWin
        ucbf_t, ucbc_t = (myValuation - ucbBid) * ucbWin, ucbBid * ucbWin

        gmpAgent.update(gmpf_t, gmpc_t, m_t[u])
        mpAgent.update(mpWin, mpf_t, mpc_t)
        ucbAgent.update(ucbWin, ucbf_t, ucbc_t, m_t[u])

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

        prova=1

    gmpCumulativePayments = np.cumsum(gmpPayments)
    mpCumulativePayments = np.cumsum(mpPayments)
    ucbCumulativePayments = np.cumsum(ucbPayments)

    winProbabilities = np.array([sum(b > m_t) / nUsers for b in possibleBids])

    ## Linear Program
    c = -(myValuation - possibleBids) * winProbabilities
    A_ub = [possibleBids * winProbabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(possibleBids))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    gamma = res.x
    expectedClairvoyantUtilities = [-res.fun for u in range(nUsers)]
    expectedClairvoyantBids = [sum(possibleBids * gamma) for u in range(nUsers)]  # (possibleBids * gamma * winProbabilities) ?


    plt.plot(m_t, markersize=1, label="Winning Bids")
    plt.plot(gmpBids, 'o', markersize=1, label="GMP Agent Bids")
    plt.plot(mpBids, 'o', markersize=1, label="MP Agent Bids")
    plt.plot(ucbBids, 'o', markersize=1, label="UCB Agent Bids")
    plt.plot(expectedClairvoyantBids, markersize=1, label="Expected Clairvoyant Bids")
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Bids$')
    plt.title('Bids history')
    plt.show()

    plt.plot(gmpCumulativePayments, label="GMP Cumulative Payments")
    plt.plot(mpCumulativePayments, label="MP Cumulative Payments")
    plt.plot(ucbCumulativePayments, label="UCB Cumulative Payments")
    plt.xlabel('$t$')
    plt.ylabel('$\sum c_t$')
    plt.axhline(B, color='red', label='Budget')
    plt.legend()
    plt.title('Cumulative Payments')
    plt.show()

    gmpCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - gmpUtilities)
    mpCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - mpUtilities)
    ucbCumulativeRegret = np.cumsum(expectedClairvoyantUtilities - ucbUtilities)
    plt.plot(gmpCumulativeRegret, label="GMP Cumulative Regret")
    plt.plot(mpCumulativeRegret, label="MP Cumulative Regret")
    plt.plot(ucbCumulativeRegret, label="UCB Cumulative Regret")
    plt.xlabel('$t$')
    plt.ylabel('$\sum R_t$')
    plt.title('Cumulative Regret of Multiplicative Pacing')
    plt.show()