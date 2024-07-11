from bidding import *
import numpy as np
import matplotlib.pyplot as plt


def biddingAdversarial(myValuation, T, numTrials, B, showPlots, seed):
    rng = np.random.RandomState(seed)

    nAdvertisers = 4
    minBid = 0.01
    maxBid = 1
    possibleBids = np.arange(minBid, myValuation, minBid)
    if len(possibleBids) == 0:
        possibleBids = np.append(possibleBids, 0)
    rho = B / T

    m_t, advertisersBids = generateRandomChangingBids(T, nAdvertisers, rng)

    '''
    colors = cm.rainbow(np.linspace(0, 1, numChanges))
    rng.shuffle(colors)
    for i in range(numChanges):
        xRange = np.arange(changingPoints[i], changingPoints[i+1])
        plt.plot(xRange, m_t[changingPoints[i]:changingPoints[i+1]], 'o', color=colors[i], markersize=1)
    plt.show()
    '''

    expectedClairvoyantUtilities, expectedClairvoyantBids = getAdversarialNonTruthfulClairvoyant(T, possibleBids, m_t, myValuation, rho)

    eta = 1 / np.sqrt(T)
    agent = FFMultiplicativePacingAgent(possibleBids, myValuation, B, T, eta, True)

    auction = GeneralizedFirstPriceAuction(1, np.ones(nAdvertisers + 1))

    utilities = np.array([])
    myBids = np.array([])
    myPayments = np.array([])
    totalWins = 0

    for u in range(T):
        # interaction
        myBid = agent.bid()
        bids = np.append(myBid, advertisersBids[:, u].ravel())
        winners, paymentsPerClick = auction.round(bids=bids)
        myWin = 0 in winners
        f_t, c_t = (myValuation - myBid) * myWin, myBid * myWin
        agent.update(myWin, f_t, c_t, m_t[u])
        # logging
        utilities = np.append(utilities, f_t)
        myBids = np.append(myBids, myBid)
        myPayments = np.append(myPayments, c_t)
        totalWins += myWin

    if showPlots:
        # %%
        '''
        for i in range(numChanges):
            xRange = np.arange(changingPoints[i], changingPoints[i+1])
            plt.plot(xRange, changing_m_t[i], 'o', color=colors[i], markersize=1)
            plt.plot(xRange, changing_expectedClairvoyantBids[i], color='black')
        plt.title('Expected Maximum Bids and (Changing) Clairvoyant Bid')
        plt.xlabel('$t$')
        plt.ylabel('$m_t$')
        plt.show()
        '''
        # %%
        plt.plot(m_t, 'o', markersize=1)
        plt.plot(expectedClairvoyantBids)
        plt.title('Expected maximum Bids and Clairvoyant Bid')
        plt.xlabel('$t$')
        plt.ylabel('$m_t$')
        plt.show()

        print(f'Total Number of Wins: {totalWins}')
        # %%
        plt.plot(myBids, 'o', markersize=1)
        plt.xlabel('$t$')
        plt.ylabel('$b_t$')
        plt.title('Chosen Bids')
        plt.show()
        # %%
        plt.plot(myBids, 'o', markersize=1)
        plt.plot(expectedClairvoyantBids)
        plt.xlabel('$t$')
        plt.ylabel('$b_t$')
        plt.title('Chosen Bids and Clairvoyants bids')
        '''
        for i in range(numChanges):
            xRange = np.arange(changingPoints[i], changingPoints[i+1])
            plt.plot(xRange, changing_expectedClairvoyantBids[i], color='black')
        '''
        plt.show()
        # %%
        cumulativePayments = np.cumsum(myPayments)
        plt.plot(cumulativePayments)
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
        '''
        flattenedCECU = [item for sublist in changing_expectedClairvoyantUtilities for item in sublist]
        CECU = np.array(flattenedCECU).ravel()
        changingCumulativeRegret = np.cumsum(CECU - utilities)
        plt.plot(changingCumulativeRegret)
        plt.xlabel('$t$')
        plt.ylabel('$\sum R_t$')
        plt.title('Cumulative Regret of Multiplicative Pacing (Changing Clairvoyant) ')
        plt.show()
        '''
    return totalWins


if __name__ == '__main__':
    biddingAdversarial(0.8, 10000, 1, 2000, True, 1)
