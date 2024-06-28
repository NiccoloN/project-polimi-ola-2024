import numpy as np

from bidding import *


def bidding(myValuation, T, numTrials, budget, showPlots):
    np.random.seed(0)

    # Auction settings
    nAdvertisers = 4
    clickThroughRates = np.ones(nAdvertisers)
    minBid = 0.01
    maxBid = 1

    # Competitors
    othersValuation = 0.6
    # if I use a really low mean bid for the competitor, the MPAgent performs better than the clairvoyant for most of the time!
    otherBids = np.clip(np.random.normal(othersValuation, 0.1, size=(nAdvertisers - 1, T)), minBid, maxBid)
    highestBids = otherBids.max(axis=0)

    highestBidsStd = np.std(highestBids)

    # Clairvoyant agent
    clairvoyantBidding, clairvoyantUtilities, clairvoyantPayments = getTruthfulClairvoyant(budget, myValuation,
                                                                                           highestBids, T)

    # Start the auction
    auction = SecondPriceAuction(clickThroughRates)

    mpUtilityHistArray = np.zeros((numTrials, T))
    mpBudgetHistArray = np.zeros((numTrials, T))
    mpRegretArray = np.zeros((numTrials, T))
    ucbUtilityHistArray = np.zeros((numTrials, T))
    ucbBudgetHistArray = np.zeros((numTrials, T))
    ucbRegretArray = np.zeros((numTrials, T))

    for trial in range(numTrials):
        np.random.seed(trial)

        # Multiplicative pacing agent
        mpAgent = MultiplicativePacingAgent(myValuation, budget, T)

        # UCB like agent
        ucbAgent = UCBAgent(budget, np.arange(minBid, myValuation, minBid), T, 0.016)

        # Auction for MP agent
        for aucInd in range(T):
            myBid = mpAgent.bid()
            bids_t = np.append(myBid, otherBids[:, aucInd].ravel())
            winner_t, payment_t = auction.round(bids_t)
            myWin = int(winner_t == 0)
            myUtility = (myValuation - highestBids[aucInd]) * myWin
            myPayment = highestBids[aucInd] * myWin
            mpAgent.update(myWin, myUtility, myPayment)

        mpWins, mpBidHist, mpUtilityHist, mpBudgetHist = mpAgent.returnHistory()
        mpUtilityHistArray[trial, :] = mpUtilityHist
        mpBudgetHistArray[trial, :] = mpBudgetHist
        mpRegretArray[trial, :] = np.cumsum(clairvoyantUtilities - mpUtilityHist)

        # Auction for UCB agent
        for aucInd in range(T):
            myBid = ucbAgent.bid()
            bids_t = np.append(myBid, otherBids[:, aucInd].ravel())
            winner_t, payment_t = auction.round(bids_t)
            myWin = int(winner_t == 0)
            myUtility = (myValuation - myBid) * myWin
            myPayment = myBid * myWin
            ucbAgent.update(myWin, myUtility, myPayment, bids_t[winner_t])

        ucbWins, ucbBidHist, ucbUtilityHist, ucbBudgetHist = ucbAgent.returnHistory()
        ucbUtilityHistArray[trial, :] = ucbUtilityHist
        ucbBudgetHistArray[trial, :] = ucbBudgetHist
        ucbRegretArray[trial, :] = np.cumsum(clairvoyantUtilities - ucbUtilityHist)

        if showPlots:
            plt.plot(highestBids, label="Other Bids")
            plt.plot(mpBidHist, label="MP Agent Bids")
            plt.plot(ucbBidHist, label="UCB Agent Bids")
            plt.legend()
            plt.xlabel('$t$')
            plt.ylabel('$Bids$')
            plt.title('Bids history')
            plt.show()

    # Plots
    if showPlots:
        plt.plot(np.cumsum(clairvoyantUtilities), label='Clairvoyant utilities')
        plt.plot(np.cumsum(mpUtilityHistArray.mean(0)), label='MP Agent utilities')
        plt.plot(np.cumsum(ucbUtilityHistArray.mean(0)), label='UCB Agent utilities')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$Cumulative utility$')
        plt.title('Utility history')
        plt.show()

        plt.plot(budget - np.cumsum(clairvoyantPayments), label='Clairvoyant Budget')
        plt.plot(mpBudgetHistArray.mean(0), label='MP Agent Budget')
        plt.plot(ucbBudgetHistArray.mean(0), label='UCB Agent Budget')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$Budget$')
        plt.title('Budget history')
        plt.show()

        plt.plot(np.cumsum(clairvoyantUtilities - mpUtilityHistArray.mean(0)), label='MP Agent Regret')
        plt.plot(np.cumsum(clairvoyantUtilities - ucbUtilityHistArray.mean(0)), label='UCB Agent Regret')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('Regret')
        plt.title('Cumulative regret')
        plt.show()

    return mpWins


if __name__ == '__main__':
    bidding(0.8, 1000, 10, 200, True)
