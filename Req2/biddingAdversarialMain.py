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
    numChanges = 10
    rho = B/T

    m_t, advertisersBids, changingPoints, check = generateRandomChangingBids(minBid, maxBid, numBids, T, numChanges+1, nAdvertisers)
    m_t = minBid + (myValuation - minBid) * (m_t - abs(min(m_t))) / (max(m_t) - abs(min(m_t)))
    for advertiser in range(nAdvertisers):
        advertisersBids[advertiser,:] = (advertisersBids[advertiser,:] + abs(min(advertisersBids[advertiser,:]))) / (max(advertisersBids[advertiser,:]) + abs(min(advertisersBids[advertiser,:])))
        prova = 1

    t = np.arange(T)
    #print(m_t)
    #print(check)

    colors = cm.rainbow(np.linspace(0, 1, numChanges))
    np.random.shuffle(colors)
    for i in range(numChanges):
        xRange = np.arange(changingPoints[i], changingPoints[i+1])
        plt.plot(xRange, m_t[changingPoints[i]:changingPoints[i+1]], 'o', color=colors[i], markersize=1)
    plt.show()

    winProbabilities = np.array([sum(b > m_t) / nUsers for b in bids])

    ## Linear Program
    c = -(myValuation - bids) * winProbabilities
    A_ub = [bids * winProbabilities]
    b_ub = [rho]
    A_eq = [np.ones(len(bids))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
    gamma = res.x
    expectedClairvoyantUtilities = [-res.fun for u in range(nUsers)]
    expectedClairvoyantBids = [sum(bids * gamma) for u in range(nUsers)]

    changing_winProbabilities = []
    changing_nUsers = []
    changing_m_t = []
    changing_expectedClairvoyantUtilities = []
    changing_expectedClairvoyantBids = []
    for change in range(numChanges):
        prova = 1
        changing_m_t.append(m_t[changingPoints[change]:changingPoints[change + 1]])
        changing_nUsers.append(changingPoints[change + 1] - changingPoints[change])
        changing_winProbabilities.append(np.array([sum(b > changing_m_t[change]) / changing_nUsers[change] for b in bids]))

        c = -(myValuation - bids) * changing_winProbabilities[change]
        A_ub = [bids * changing_winProbabilities[change]]
        b_ub = [rho]
        A_eq = [np.ones(len(bids))]
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        gamma = res.x
        changing_expectedClairvoyantUtilities.append([-res.fun for u in range(changing_nUsers[change])])
        changing_expectedClairvoyantBids.append([sum(bids * gamma) for u in range(changing_nUsers[change])])


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

    # %%
    for i in range(numChanges):
        xRange = np.arange(changingPoints[i], changingPoints[i+1])
        plt.plot(xRange, changing_m_t[i], 'o', color=colors[i], markersize=1)
        plt.plot(xRange, changing_expectedClairvoyantBids[i], color='black')
    plt.title('Expected Maximum Bids and (Changing) Clairvoyant Bid')
    plt.xlabel('$t$')
    plt.ylabel('$m_t$')
    plt.show()
    # %%
    plt.plot(m_t)
    plt.plot(expectedClairvoyantBids)
    plt.title('Expected maximum Bids and Clairvoyant Bid')
    plt.xlabel('$t$')
    plt.ylabel('$m_t$')
    plt.show()

    print(f'Total Number of Wins: {totalWins}')
    # %%
    plt.plot(myBids)
    plt.plot(expectedClairvoyantBids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Chosen Bids')
    plt.show()
    # %%
    plt.plot(myBids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Chosen Bids and Clairvoyants bids')
    plt.plot(expectedClairvoyantBids)
    for i in range(numChanges):
        xRange = np.arange(changingPoints[i], changingPoints[i+1])
        plt.plot(xRange, changing_expectedClairvoyantBids[i], color='black')
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
    cumulativeRegret = np.cumsum(expectedClairvoyantUtilities - utilities)
    plt.plot(cumulativeRegret)
    plt.xlabel('$t$')
    plt.ylabel('$\sum R_t$')
    plt.title('Cumulative Regret of Multiplicative Pacing')
    plt.show()
    # %%
    flattenedCECU = [item for sublist in changing_expectedClairvoyantUtilities for item in sublist]
    CECU = np.array(flattenedCECU).ravel()
    changingCumulativeRegret = np.cumsum(CECU - utilities)
    plt.plot(changingCumulativeRegret)
    plt.xlabel('$t$')
    plt.ylabel('$\sum R_t$')
    plt.title('Cumulative Regret of Multiplicative Pacing (Changing Clairvoyant) ')
    plt.show()

    prova = 2