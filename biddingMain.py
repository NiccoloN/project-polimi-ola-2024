import matplotlib.pyplot as plt

from bidding import *

if __name__ == '__main__':
    # Auction settings
    nAdvertisers = 4
    clickThroughRates = np.ones(nAdvertisers)
    myValuation = 0.8
    budget = 100
    nRounds = 1000
    minBid = 0.01
    maxBid = 1

    # Competitors
    othersValuation = 0.6
    # if I use a really low mean bid for the competitor, the MPAgent performs better than the clairvoyant for most of the time!
    otherBids = np.clip(np.random.normal(othersValuation, 0.1, size=(nAdvertisers - 1, nRounds)), minBid, maxBid)
    highestBids = otherBids.max(axis=0)

    # Clairvoyant agent
    clairvoyantBidding, clairvoyantUtilities, clairvoyantPayments = getDeterministicClairvoyant(budget, myValuation, highestBids, nRounds)

    # Multiplicative pacing agent
    mpAgent = MultiplicativePacingAgent(myValuation, budget, nRounds)

    # Start the auction
    auction = SecondPriceAuction(clickThroughRates)
    for aucInd in range(nRounds):
        myBid = mpAgent.bid()
        bids_t = np.append(myBid, otherBids[:, aucInd].ravel())
        winner_t, payment_t = auction.round(bids_t)
        myWin = int(winner_t == 0)
        myUtility = (myValuation - highestBids[aucInd]) * myWin
        myPayment = highestBids[aucInd] * myWin
        mpAgent.update(myWin, myUtility, myPayment)

    myWins, myBidHist, myUtilityHist, myBudgetHist = mpAgent.returnHistory()

    # Plots
    plt.plot(highestBids, label="Other Bids")
    plt.plot(myBidHist, label="MP Agent Bids")
    # plt.plot(clairvoyantPayments+minBid, label="Clairvoyant bids")
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Bids$')
    plt.title('Bids history')
    plt.show()

    plt.plot(np.cumsum(myUtilityHist), label='MP Agent utilities')
    plt.plot(np.cumsum(clairvoyantUtilities), label='Clairvoyant utilities')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Cumulative utility$')
    plt.title('Utility history')
    plt.show()

    plt.plot(myBudgetHist, label='MP Agent Budget')
    plt.plot(budget-np.cumsum(clairvoyantPayments), label='Clairvoyant Budget')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$Budget$')
    plt.title('Budget history')
    plt.show()

    plt.plot(np.cumsum(clairvoyantUtilities-myUtilityHist), label='MP Agent Regret')
    plt.xlabel('$t$')
    plt.ylabel('Regret')
    plt.title('Cumulative regret of MP Agent')
    plt.show()

