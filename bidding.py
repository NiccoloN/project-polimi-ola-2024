import numpy as np
from matplotlib import pyplot as plt


class Auction:
    def __init__(self, *args, **kwargs):
        pass

    def get_winners(self, bids):
        pass

    def get_payments_per_click(self, winners, values, bids):
        pass

    def round(self, bids):
        winners, values = self.get_winners(bids) # allocation mechanism!
        payments_per_click = self.get_payments_per_click(winners, values, bids)
        return winners, payments_per_click


class SecondPriceAuction(Auction):
    def __init__(self, ctrs):
        self.ctrs = ctrs
        self.n_adv = len(self.ctrs)

    def get_winners(self, bids):
        adv_values = self.ctrs * bids
        adv_ranking = np.argsort(adv_values)
        winner = adv_ranking[-1]
        return winner, adv_values

    def get_payments_per_click(self, winners, values, bids):
        adv_ranking = np.argsort(values)
        second = adv_ranking[-2]
        payment = values[second] / self.ctrs[winners]
        return payment.round(2)


class VCGAuction(Auction):
    def __init__(self, ctrs, lambdas):
        self.ctrs = ctrs
        self.lambdas = lambdas
        self.n_adv = len(self.ctrs)
        self.n_slots = len(self.lambdas)

    def get_winners(self, bids):
        adv_values = self.ctrs * bids
        adv_ranking = np.argsort(adv_values)
        winners = adv_ranking[-self.n_slots:]
        winners_values = adv_values[winners]
        return winners, winners_values

    def get_payments_per_click(self, winners, values, bids):
        payments_per_click = np.zeros(self.n_slots)
        for i, w in enumerate(winners):
            Y = sum(np.delete(values, i) * self.lambdas[-self.n_slots + 1:])
            X = sum(np.delete(values * self.lambdas, i))
            payments_per_click[i] = (Y - X) / (self.lambdas[i] * self.ctrs[w])
        return payments_per_click.round(2)


def get_clairvoyant_truthful(B, my_valuation, m_t, n_users):
    utility = (my_valuation - m_t) * (my_valuation >= m_t)
    sorted_round_utility = np.flip(np.argsort(utility))
    clairvoyant_utilities = np.zeros(n_users)
    clairvoyant_bids = np.zeros(n_users)
    clairvoyant_payments = np.zeros(n_users)
    c = 0
    i = 0
    while c <= B - 1 and i < n_users:
        clairvoyant_bids[sorted_round_utility[i]] = 1
        clairvoyant_utilities[sorted_round_utility[i]] = utility[sorted_round_utility[i]]
        clairvoyant_payments[sorted_round_utility[i]] = m_t[sorted_round_utility[i]]
        c += m_t[sorted_round_utility[i]]
        i += 1
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments
