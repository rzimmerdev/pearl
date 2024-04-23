import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


class HMM:
    def __init__(self, hidden_states, observation_space, d):
        self.hidden_states = hidden_states
        self.observation_space = observation_space
        self.d = d
        self.transition_matrix = np.random.rand(hidden_states, hidden_states)
        self.emission_matrix = np.random.rand(hidden_states, observation_space)

        self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:, np.newaxis]
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1)[:, np.newaxis]

    def forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.hidden_states))

        # Initialization
        alpha[0] = self.emission_matrix[:, observation_to_index(observations[0])]

        # Induction
        for t in range(1, T):
            for j in range(self.hidden_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition_matrix[:, j]) * self.emission_matrix[j, observation_to_index(observations[t])]

        return alpha

    def backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.hidden_states))

        # Initialization
        beta[-1] = np.ones(self.hidden_states)

        # Induction
        for t in range(T - 2, -1, -1):
            for i in range(self.hidden_states):
                beta[t, i] = np.sum(self.transition_matrix[i] * self.emission_matrix[:, observation_to_index(observations[t + 1])] * beta[t + 1])

        return beta


observation_space = np.linspace(-2, 2, 100)


def to_observation_space(x):
    return observation_space[np.argmin(np.abs(observation_space - x))]


def observation_to_index(x):
    return np.argmin(np.abs(observation_space - x))


def mixture_sampling(transition_matrix, emission_matrix, initial_state, T):
    current_state = 0
    observations = []

    for t in range(T):
        current_state = np.random.choice(np.arange(len(transition_matrix)), p=transition_matrix[current_state])
        observation = np.random.choice(observation_space, p=emission_matrix[current_state])

        observations.append(observation)

    return observations


hidden_states = 3
# d = 1

true_transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.7, 0.2],
    [0.2, 0.1, 0.7]
])

emissions = [
    (0, 1),
    (1, 1),
    (-1, 2)
]

true_emission_matrix = np.array([
    [norm(loc=mean, scale=std).pdf(observation_space) for mean, std in emissions]
])[0]

true_emission_matrix /= np.sum(true_emission_matrix, axis=1)[:, np.newaxis]

data = mixture_sampling(true_transition_matrix, true_emission_matrix, 0, 1000)

# fig = go.Figure()
# acc = np.cumsum(data)
# fig.add_trace(go.Scatter(x=np.arange(len(data)), y=acc, mode='markers'))
# fig.show()

hmm = HMM(hidden_states, len(observation_space), 1)

def baum_welch(hmm, observations, n_iter=100):
    for _ in range(n_iter):
        alpha = hmm.forward(observations)
        beta = hmm.backward(observations)

        T = len(observations)
        gamma = alpha * beta / np.sum(alpha[-1])

        xi = np.zeros((T - 1, hmm.hidden_states, hmm.hidden_states))
        for t in range(T - 1):
            for i in range(hmm.hidden_states):
                for j in range(hmm.hidden_states):
                    xi[t, i, j] = alpha[t, i] * hmm.transition_matrix[i, j] * hmm.emission_matrix[j, observation_to_index(observations[t + 1])] * beta[t + 1, j] / np.sum(alpha[-1])

        hmm.transition_matrix = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]

        for i in range(hmm.hidden_states):
            for k in range(hmm.observation_space):
                hmm.emission_matrix[i, k] = np.sum(gamma[:, i] * (observations == observation_space[k])) / np.sum(gamma[:, i])

    return hmm


hmm = baum_welch(hmm, data, n_iter=100)

print(hmm.transition_matrix)