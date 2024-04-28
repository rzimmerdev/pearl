import numpy as np
from scipy.stats import norm
from plotly import graph_objects as go

from .utils import mse, mixture_sampling_normal, mixture_sampling_any


class HMM:
    def __init__(self, states: int, space: np.ndarray, d: int):
        self.hidden_states = states
        self.observation_space = space
        self.d = d
        self.transition_matrix = np.random.rand(self.hidden_states, self.hidden_states)
        self.emission_matrix = np.random.rand(self.hidden_states, len(space))

        self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:, np.newaxis]
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1)[:, np.newaxis]

    def observation_index(self, x):
        if isinstance(x, np.ndarray):
            return np.argmin(np.abs(self.observation_space - x[:, np.newaxis]), axis=1)
        return np.argmin(np.abs(self.observation_space - x))

    def clip_observation(self, x):
        return self.observation_space[self.observation_index(x)]

    def forward(self, data):
        observation_length = len(data)

        alpha = np.zeros((observation_length, self.hidden_states))
        alpha[0] = self.emission_matrix[:, self.observation_index(data[0])]

        for t in range(1, observation_length):
            alpha[t] = (
                    np.sum(alpha[t - 1] * self.transition_matrix.T, axis=1)
                    * self.emission_matrix[:, self.observation_index(data[t])]
            )
            alpha[t] /= np.sum(alpha[t])

        return alpha

    def backward(self, data):
        observation_length = len(data)

        beta = np.zeros((observation_length, self.hidden_states))
        beta[observation_length - 1] = 1

        for t in range(observation_length - 2, -1, -1):
            beta[t] = np.sum(
                self.transition_matrix * self.emission_matrix[:, self.observation_index(data[t + 1])] * beta[t + 1],
                axis=1,
            )
            beta[t] /= np.sum(beta[t])

        return beta

    def baum_welch(self, data, iterations):
        observation_length = len(data)

        for k in range(iterations):
            alpha = self.forward(data)
            beta = self.backward(data)

            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

            xi = np.zeros((observation_length, self.hidden_states, self.hidden_states))

            for t in range(observation_length - 1):
                for i in range(self.hidden_states):
                    for j in range(self.hidden_states):
                        xi[t, i, j] = (alpha[t, i] *
                                       self.transition_matrix[i, j] *
                                       self.emission_matrix[j, self.observation_index(data[t + 1])] *
                                       beta[t + 1, j])
                xi[t] /= np.sum(xi[t])

            transition_matrix_new = np.sum(xi, axis=0) / np.sum(gamma, axis=0)[:, np.newaxis]
            emission_matrix_new = np.zeros((self.hidden_states, len(self.observation_space)))

            for i in range(self.hidden_states):
                emission_matrix_new[i] = np.sum(
                    gamma[:, i] *
                    (self.observation_index(data) == np.arange(len(self.observation_space))[:, np.newaxis]),
                    axis=1,
                )
                emission_matrix_new[i] /= np.sum(gamma[:, i])

            transition_matrix_new /= np.sum(transition_matrix_new, axis=1)[:, np.newaxis]
            emission_matrix_new /= np.sum(emission_matrix_new, axis=1)[:, np.newaxis]

            if k % 10 == 0:
                print(f"Epoch {k}, change == : {mse(self.transition_matrix, transition_matrix_new)}")

            self.transition_matrix = transition_matrix_new
            self.emission_matrix = emission_matrix_new

    def mixture_sampling(self, T):
        return mixture_sampling_any(self.transition_matrix, self.emission_matrix, self.observation_space, T)


def main():
    hidden_states = 3
    # d = 1

    true_transition_matrix = np.array(
        [[0.7, 0.2, 0.1],
         [0.1, 0.7, 0.2],
         [0.2, 0.1, 0.7]]
    )

    emissions = [(0, 0.1),
                 (2, 0.1),
                 (-2, 0.1)]

    observation_space = np.linspace(-2, 2, 100)

    true_emission_matrix = np.array([[norm(loc=mean, scale=std).pdf(observation_space) for mean, std in emissions]])[0]
    true_emission_matrix /= np.sum(true_emission_matrix, axis=1)[:, np.newaxis]

    data = mixture_sampling_normal(true_transition_matrix, true_emission_matrix, observation_space, 1000)

    hmm = HMM(hidden_states, observation_space, 1)

    print(true_transition_matrix)
    hmm.baum_welch(data, 100)
    print(np.round(hmm.transition_matrix, 2))
    print(np.round(true_transition_matrix, 2))

    acc = np.cumsum(data)
    acc_hmm = np.cumsum(hmm.mixture_sampling(1000))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=acc, mode="markers", name="True"))
    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=acc_hmm, mode="markers", name="HMM"))
    fig.show()


if __name__ == "__main__":
    main()
