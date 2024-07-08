import numpy as np
from bidding import *
from pricing import *
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize


if __name__ == '__main__':
    nAdvertisers = 4
    # clickThroughRates = np.ones(nAdvertisers)
    minBid = 0.2
    maxBid = 1
    myValuation = 0.8
    othersValuation = 0.6
    bids = np.arange(0, 1, 0.01)
    numBids = len(bids)
    # numBids = 100
    nUsers = 10000
    B = nUsers/10
    T = nUsers
    numChanges = 9
    rho = B/T

    m_t, advertisersBids, changingPoints, check = generateRandomChangingBids(minBid, maxBid, numBids, T, numChanges, nAdvertisers)
    m_t = minBid + (myValuation - minBid) * (m_t - abs(min(m_t))) / (max(m_t) - abs(min(m_t)))
    for advertiser in range(nAdvertisers):
        advertisersBids[advertiser,:] = (advertisersBids[advertiser,:] + abs(min(advertisersBids[advertiser,:]))) / (max(advertisersBids[advertiser,:]) + abs(min(advertisersBids[advertiser,:])))
        prova = 1

    t = np.arange(T)
    #print(m_t)
    #print(check)

    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(numChanges):
        xRange = np.arange(changingPoints[i-1], changingPoints[i])
        plt.plot(xRange, m_t[changingPoints[i-1]:changingPoints[i]], 'o', color=colors[i-1])

    win_probabilities = np.array([sum(b > m_t) / nUsers for b in bids])

    ## Linear Program
    c = -(myValuation - bids) * win_probabilities
    A_ub = [bids * win_probabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(bids))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    gamma = res.x
    expected_clairvoyant_utilities = [-res.fun for u in range(nUsers)]
    expected_clairvoyant_bids = [sum(bids * gamma) for u in range(nUsers)]


    eta = 1 / np.sqrt(nUsers)
    agent = FFMultiplicativePacingAgent(bids_set=bids,
                                        valuation=myValuation,
                                        budget=B,
                                        T=nUsers,
                                        eta=eta)

    auction = FirstPriceAuction(np.ones(nAdvertisers + 1))

    utilities = np.array([])
    myBids = np.array([])
    myPayments = np.array([])
    totalWins = 0

    np.random.seed(T)
    for u in range(nUsers):
        #if u >= advertisersBids.shape[1]:
            #break
        # interaction
        myBid = agent.bid()
        bids = np.append(myBid, advertisersBids[:, u].ravel())
        winners, paymentsPerClick = auction.round(bids=bids)
        myWin = int(winners == 0)
        f_t, c_t = (myValuation - myBid) * myWin, myBid * myWin
        agent.update(f_t, c_t, m_t[u])
        # logging
        utilities = np.append(utilities, f_t)
        myBids = np.append(myBids, myBid)
        myPayments = np.append(myPayments, c_t)
        totalWins += myWin


    plt.plot(m_t)
    plt.plot(expected_clairvoyant_bids)
    plt.title('Expected maximum bid')
    plt.xlabel('$t$')
    plt.ylabel('$m_t$')
    plt.show()

    print(f'Total # of Wins: {totalWins}')
    # %%
    plt.plot(myBids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Chosen Bids')
    plt.plot(expected_clairvoyant_bids)
    plt.show()
    # %%
    cumulative_payments = np.cumsum(myPayments)
    plt.plot(cumulative_payments)
    plt.xlabel('$t$')
    plt.ylabel('$\sum c_t$')
    plt.axhline(B, color='red', label='Budget')
    plt.legend()
    plt.title('Cumulative Payments of Multiplicative Pacing')
    plt.show()
    # %%
    #cumulative_regret = np.cumsum(expected_clairvoyant_utilities[0:min(advertisersBids.shape[1], nUsers)] - utilities) #modificato (male)
    cumulative_regret = np.cumsum(expected_clairvoyant_utilities - utilities)
    plt.plot(cumulative_regret)
    plt.xlabel('$t$')
    plt.ylabel('$\sum R_t$')
    plt.title('Cumulative Regret of Multiplicative Pacing')
    plt.show()

    prova = 2