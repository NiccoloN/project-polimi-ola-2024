# %% md
# **Adversarial Settings**
# %%
import numpy as np
import matplotlib.pyplot as plt


# %% md
### In Adversarial Settings we generalize the process generating rewards. Instead of being sampled from a probability distribution, rewards are chosen by an adversary that can observe the bandit algorithm is facing.
# %% md
## **The Adversarial Expert Setting**
# %% md
### In Expert settings, at each round, an agent chooses an arm and incurs its loss, but can observe the losses of all the other arms.
# %% md
### Thus, there is no exploration-exploitation trade-off dilemma, since exploration is pointless when all information is available.
# %%
class AdversarialExpertEnvironment:
    # unlike stochastic environment no need for parameters of probability distribution:
    # it doesn't compute all the sequences of losses, instead we provide the environment all those sequences
    # it doesn't compute rewards, just returns the right loss depending on the time
    def __init__(self, loss_sequence):
        self.loss_sequence = loss_sequence
        self.t = 0 # time initialized

    # action isn't provided: in every round, independently of the action,
    # return the whole array of losses of the specific time. That's because we're in an expert setting:
    # learner observes losses associated to every arm at each time step (easier than bandit setting)
    def round(self):  # we do not need to receive a specific arm
        l_t = self.loss_sequence[self.t, :]  ## we return the whole loss vector
        self.t += 1
        return l_t


# %%
# I see the algorithm, then decide an arbitrary sequence of rewards:
loss_seq = np.array(
    [[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
loss_seq.shape
# %%
env = AdversarialExpertEnvironment(loss_seq)
env.t, env.round(), env.t
# %% md
### Of course, the Adversarial setting generalizes the Stochastic setting, since the adversary is free to choose to sample the rewards from a probability distribution, thus any Stochastic MAB is also a special instance of Adversarial MAB.
# %% md
### The notion of regret change its meaning, the clairvoyant is not the agent always choosing the best action at any moment, but the agent always pulling the arm having the best cumulative reward at the end.
# %%
print(f'Best achievable cumulative loss: {loss_seq.min(axis=1).sum()}')
print(f'Best achievable cumulative loss when always pulling the same arm: {loss_seq.sum(axis=0).min()}')
# %% md
### We will compare to the second.
# %% md
### We need to define a new clairvoyant: _Best arm in hindsight_.
# %%
best_arm = np.argmin(loss_seq.sum(axis=0))
print(f'The best arm in hindsight is {best_arm}')
# %%
clairvoyant_losses = loss_seq[:, best_arm]
clairvoyant_losses


# %% md
## **The Hedge algorithm**
# it produces a probability distribution over the actions (randomized algorithm), keeps track of the probability
# of choosing each arm, at each round chooses an arm sampling from this probability distribution,
# observes the loss associated to all arms, suffers an expected loss (expected loss wrt randomness of algorithm =
# single loss of chosen arm), update weights using an exponential rule (the more is the loss given by action a,
# the more the probability of choosing that action is penalized)
# notion of regret is changed: not compared to clairvoyant (would be too strong: it would know before each round
# all losses), but to the best arm in hindsight instead. This is the arm whose sum of losses is lowest (line 110)
# %%
class HedgeAgent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate # optimal learning rate: sqrt(logK / T) (see theory)
        self.weights = np.ones(K) # weights initialized to 1
        self.x_t = np.ones(K) / K # probability distribution: normalized weights
        self.a_t = None # action chosen
        self.t = 0 # time initialized

    def pull_arm(self): # whenever I pull an arm I just perform a sample
        self.x_t = self.weights / sum(self.weights) # compute probability distribution as normalized weights
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t) # obtain an arm by performing random choice (computed probabilities)
        return self.a_t # return the action

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate * l_t) # update vector of weights by means of the vector of losses
        self.t += 1 # increment time


# %% md
### Due to practical reasons, we generate the losses using a probability distribution. However, in principle, we would be free to choose any sequence we'd like.
# %%
T = 10000
K = 3
loss_seq = np.zeros((T, K))
np.random.seed(17)
loss_seq[:, 0] = np.random.binomial(n=1, p=0.7, size=T)
loss_seq[:, 1] = np.random.binomial(n=1, p=0.5, size=T)
loss_seq[:, 2] = np.random.binomial(n=1, p=0.25, size=T)
# %% md
### We distinguish two ways to computed regret: actual regret and regret in expectation.
# %% md
### **Important Remark**: in adversarial settings, the expectation on regret is only taken w.r.t. to the randomness of the algorithm, since by definition the sequence of losses is not stochastic.
# %%
# set up a simulation:
learning_rate = np.sqrt(np.log(K) / T)  #  we set the learning rate as prescribed by the theory

## agent = HedgeAgent(K, learning_rate) # define an agent
## env = AdversarialExpertEnvironment(loss_seq) # define an environment
# commented out since later included in a for loop

best_arm = np.argmin(loss_seq.sum(axis=0)) # obtain best arm in hindsight
clairvoyant_losses = loss_seq[:, best_arm] # compute array of losses incurred by clairvoyant (ex [0,1,1,1,0,0])

# we log to different cumulative losses for the agent
## agent_losses = np.array([]) # store agent losses
## expected_agent_losses = np.array([]) # store expected agent losses
# commented out since later included in a for loop


# we're now differentiating two notions of regret: the regret in expectation (I observe all the losses,
# I can compute what was the expectation, because I know which are the probabilities associated to every arm);
# but we also compute the losses by using the actual single loss from the chosen arm (this just to show that
# the actual from a single arm goes in expectation over multiple trials like the expected agent loss)

all_cumulative_regret = []
for seed in range(10):
    agent = HedgeAgent(K, learning_rate) # after each trial use a different Hedge agent
    env = AdversarialExpertEnvironment(loss_seq) # after each trial use a different using the SAME sequence of losses
    agent_losses = np.array([])  # store agent losses
    expected_agent_losses = np.array([]) # store expected agent losses
    for t in range(T):
        a_t = agent.pull_arm()
        l_t = env.round()
        agent.update(l_t)
        # logging
        agent_losses = np.append(agent_losses, l_t[a_t])
        expected_agent_losses = np.append(expected_agent_losses, np.dot(l_t, agent.x_t))
    all_cumulative_regret.append(np.cumsum(agent_losses - clairvoyant_losses))

plt.plot(np.array(all_cumulative_regret).mean(axis=0), label='Estimated Regret')
plt.plot(np.cumsum(agent_losses - clairvoyant_losses), label='Actual Regret')
# multiple trials (same environment) -> average of actual losses = expected loss
# this probability distribution is not due to the environment, but to the agent randomization (always same sequence of losses)
# very important: the only source of randomness is the algorithm (not evaluated environment uncertainty, just agent uncertainty)
# note that in an expert setting, computing actual loss over multiple trials and averaging is unnecessary:
# expected loss can be computed in closed form by multiplying the loss vector w/ the probability vector over the arms
plt.plot(np.cumsum(expected_agent_losses - clairvoyant_losses), label='Expected Regret')
plt.title('Cumulative Regret of Hedge in Adversarial Expert Setting')
plt.xlabel('$t$')
plt.legend()
plt.show()
# %%
print(f'Best arm in hindsight: {best_arm}')
print(f'Final allocation :{agent.x_t}')  # the best arm is the one having more weight
# %%
print(f'Theoretical bound: {2 * np.sqrt(T * np.log(K))}')
print(f'Actual Total Regret {sum(agent_losses) - sum(clairvoyant_losses)}')
print(f'Expected Total Regret {sum(expected_agent_losses) - sum(clairvoyant_losses)}')


# %% md
## **The Adversarial Bandit Setting**
# %% md
### In Bandit Settings, the agent chooses one arm and can only observe the loss associated to that single arm.
# Thus, the feedback received from the environment is limited. This problem is intrinsically harder than the
# expert setting, due to the more limited information available.
# %%
class AdversarialBanditEnvironment:
    def __init__(self, loss_sequence):
        self.loss_sequence = loss_sequence
        self.t = 0

    def round(self, a_t):  # we need to receive a specific arm
        l_t = self.loss_sequence[self.t, a_t]  ## we return only the loss corresponding to the chosen arm
        self.t += 1
        return l_t


# %% md
### As before, the sequence of losses is arbitrary.
# %%
# I see the algorithm, then decide the sequence of rewards:
loss_seq = np.array(
    [[0, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
env = AdversarialBanditEnvironment(loss_seq)
env.t, env.round(0), env.t
# %% md
### As before, the regret is computed w.r.t. the best arm in hindsight
# %%
print(f'Best achievable cumulative loss: {loss_seq.min(axis=1).sum()}')
print(f'Best achievable cumulative loss when always pulling the same arm: {loss_seq.sum(axis=0).min()}')


# %% md
### We will compare to the second.
# %% md
### However, the problem is now harder, since we can only observe the feedback associated to the arm we choose. Thus, we need to deal with the exploration-exploitation trade-off.
# %% md
## **Extending Hedge to a Bandit Setting: the EXP3 Algorithm**
# in Hedge each action was penalized only based on the loss obtained from that action, here I'm dividing
# the loss by the probability of choosing that action (see theory) -> more penalization for bad arms
# %%
class EXP3Agent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K) / K
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        l_t_tilde = l_t / self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp(-self.learning_rate * l_t_tilde)
        self.N_pulls[self.a_t] += 1
        self.t += 1


# %%
T = 10000
K = 3
loss_seq = np.zeros((T, K))
np.random.seed(17)
loss_seq[:, 0] = np.random.binomial(n=1, p=0.7, size=T)
loss_seq[:, 1] = np.random.binomial(n=1, p=0.5, size=T)
loss_seq[:, 2] = np.random.binomial(n=1, p=0.25, size=T)
# %% md
### Again, we can compute the received losses in two ways: actual and expected. Note that in bandit settings, differently from experts, the expected loss cannot be computed in practice, since it would require to know all the losses associated to every arm.
# %%
learning_rate = np.sqrt(
    np.log(K) / (K * T))  #  we set the learning rate as prescribed by the theory (different from expert setting)

agent = EXP3Agent(K, learning_rate)
env = AdversarialBanditEnvironment(loss_seq)

best_arm = np.argmin(loss_seq.sum(axis=0))
clairvoyant_losses = loss_seq[:, best_arm]

agent_losses = np.array([])
expected_agent_losses = np.array([])
for t in range(T):
    a_t = agent.pull_arm()
    l_t = env.round(a_t)
    agent.update(l_t)
    # logging
    agent_losses = np.append(agent_losses, l_t)
    expected_agent_losses = np.append(expected_agent_losses,
                                      np.dot(agent.x_t,
                                             env.loss_sequence[t - 1, :]))

plt.plot(np.cumsum(agent_losses - clairvoyant_losses), label='Actual Regret')
plt.plot(np.cumsum(expected_agent_losses - clairvoyant_losses), label='Expected Regret')
# the expected regret is unknown to the algorithm (can be computed in close form w/ sequence of losses)
plt.title('Cumulative Regret of EXP3 in Bandit Setting')
plt.xlabel('$t$')
plt.legend()
plt.show()
# %% md
### Remark: the agent only observes the loss associated to the chosen arm, expected loss cannot be computed without knowing the losses from all arms (like in the expert setting). Thus, the algorithm only reacts based on the actual loss incurred.
# %%
print(f'Best arm in hindsight: {best_arm}')
print(f'Final allocation :{agent.x_t}')  # the best arm is the one having more weight
# %%
print(
    f'Theoretical bound: {2 * np.sqrt(T * K)}')  # The theoretical bound has worsen dependence on K w.r.t. expert setting (from log(K) to sqrt(K))
print(f'Actual Total Regret {sum(agent_losses) - sum(clairvoyant_losses)}')
# %% md
### We can quantify the uncertainty on expected regret, where the only uncertainty source is algorithm's randomization.
# %%
learning_rate = np.sqrt(
    np.log(K) / (K * T))  #  we set the learning rate as prescribed by the theory (different from expert setting)

best_arm = np.argmin(loss_seq.sum(axis=0))
clairvoyant_losses = loss_seq[:, best_arm]

n_trials = 20

exp3_regret_per_trial = []
# we keep the loss sequence fixed, we will only observe uncertainty due to algorithm's randomizations
for trial in range(n_trials): # as for Hedge, multiple trials just to estimate the uncertainty
    agent = EXP3Agent(K, learning_rate)
    env = AdversarialBanditEnvironment(loss_seq)

    expected_agent_losses = np.array([])
    for t in range(T):
        a_t = agent.pull_arm()
        l_t = env.round(a_t)
        agent.update(l_t)
        # logging
        expected_agent_losses = np.append(expected_agent_losses,
                                          np.dot(agent.x_t,
                                                 env.loss_sequence[t - 1, :]))

    cumulative_regret = np.cumsum(expected_agent_losses - clairvoyant_losses)
    exp3_regret_per_trial.append(cumulative_regret)

exp3_regret_per_trial = np.array(exp3_regret_per_trial)

exp3_average_regret = exp3_regret_per_trial.mean(axis=0)
exp3_regret_sd = exp3_regret_per_trial.std(axis=0)

plt.plot(np.arange(T), exp3_average_regret, label='EXP3')
plt.title('Cumulative Regret of EXP3') # same line as "expected regret" from previous plot
plt.fill_between(np.arange(T),
                 exp3_average_regret - exp3_regret_sd / np.sqrt(n_trials),
                 exp3_average_regret + exp3_regret_sd / np.sqrt(n_trials),
                 alpha=0.3)
plt.xlabel('$t$')
plt.legend()
plt.show()

plt.barh(y=['0', '1', '2'], width=agent.N_pulls)
plt.title('Number of pulls per arm of EXP3')
plt.ylabel('$a$')
plt.xlabel('$N_T(a)$')
plt.show()