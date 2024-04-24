import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import plotly.graph_objects as go


class HMM:
    def __init__(self, states, space, d):
        self.hidden_states = states
        self.observation_space = space
        self.d = d
        self.transition_matrix = np.random.rand(hidden_states, hidden_states)
        self.emission_matrix = np.random.rand(hidden_states, len(space))

        self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:, np.newaxis]
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1)[:, np.newaxis]

    def observation_index(self, x):
        if isinstance(x, np.ndarray):
            return np.argmin(np.abs(self.observation_space - x[:, np.newaxis]), axis=1)
        return np.argmin(np.abs(self.observation_space - x))

    def clip_observation(self, x):
        return self.observation_space[self.observation_index(x)]

    def forward(self, data):
        T = len(data)
        alpha = np.zeros((T, self.hidden_states))

        # Initialization
        alpha[0] = self.emission_matrix[:, self.observation_index(data[0])]

        # Recursion
        for t in range(1, T):
            for i in range(self.hidden_states):
                alpha[t, i] = np.sum(alpha[t - 1] * self.transition_matrix[:, i]) * self.emission_matrix[i, self.observation_index(data[t])]

            # Normalize
            alpha[t] /= np.sum(alpha[t])

        return alpha

    def backward(self, data):
        T = len(data)
        beta = np.zeros((T, self.hidden_states))

        # Initialization
        beta[T - 1] = 1

        # Recursion
        for t in range(T - 2, -1, -1):
            for i in range(self.hidden_states):
                beta[t, i] = np.sum(self.transition_matrix[i, :] * self.emission_matrix[:, self.observation_index(data[t + 1])] * beta[t + 1, :])

            # Normalize
            beta[t] /= np.sum(beta[t])

        return beta

    def baum_welch(self, data, iterations):
        T = len(data)
        for i in tqdm(range(iterations)):
            alpha = self.forward(data)
            beta = self.backward(data)

            # E-step
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

            xi = np.zeros((T, self.hidden_states, self.hidden_states))
            for t in range(T - 1):
                for i in range(self.hidden_states):
                    for j in range(self.hidden_states):
                        xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * self.emission_matrix[j, self.observation_index(data[t + 1])] * beta[t + 1, j]
                xi[t] /= np.sum(xi[t])

            # M-step
            self.transition_matrix = np.sum(xi, axis=0) / np.sum(gamma, axis=0)[:, np.newaxis]
            self.emission_matrix = np.zeros((self.hidden_states, len(self.observation_space)))
            for i in range(self.hidden_states):
                for j in range(len(self.observation_space)):
                    self.emission_matrix[i, j] = np.sum(gamma[:, i] * (self.observation_index(data) == j)) / np.sum(gamma[:, i])

            self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:, np.newaxis]
            self.emission_matrix /= np.sum(self.emission_matrix, axis=1)[:, np.newaxis]


def mixture_sampling(transition_matrix, emission_matrix, space, T):
    current_state = 0
    observations = []

    for t in range(T):
        current_state = np.random.choice(np.arange(len(transition_matrix)), p=transition_matrix[current_state])
        observation = np.random.choice(space, p=emission_matrix[current_state])

        observations.append(observation)

    return np.array(observations)


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

observation_space = np.linspace(-2, 2, 100)

true_emission_matrix = np.array([
    [norm(loc=mean, scale=std).pdf(observation_space) for mean, std in emissions]
])[0]

true_emission_matrix /= np.sum(true_emission_matrix, axis=1)[:, np.newaxis]

data = mixture_sampling(true_transition_matrix, true_emission_matrix, observation_space, 1000)

# fig = go.Figure()
# acc = np.cumsum(data)
# fig.add_trace(go.Scatter(x=np.arange(len(data)), y=acc, mode='markers'))
# fig.show()

hmm = HMM(hidden_states, observation_space, 1)
alpha = hmm.forward(data)
beta = hmm.backward(data)
print(alpha.shape)  # (T, hidden_states)
print(beta.shape)  # (T, hidden_states)

hmm.baum_welch(data, 50)

# round transition matrix to 2 decimal places
print(np.round(hmm.transition_matrix, 2))

print(true_transition_matrix)
print(np.allclose(
    np.sum(hmm.transition_matrix, axis=1),
    np.ones(hidden_states)
))
