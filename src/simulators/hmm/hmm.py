from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from plotly import graph_objects as go


class MSE:
    def __call__(self, x, y):
        return np.mean((x - y) ** 2)


class TransitionMatrix:
    def __init__(self, states: int):
        self.states = states
        self.matrix = np.random.rand(self.states, self.states)

        self.matrix /= np.sum(self.matrix, axis=1)[:, np.newaxis]

    def __call__(self, *args, **kwargs):
        return self.matrix

    def p(self, state):
        return self.matrix[state]

    def transition(self, state):
        return np.random.choice(np.arange(self.states), p=self.p(state))


class EmissionMatrix:
    def __init__(self, states: int, spaces: Dict[str, np.ndarray]):
        self.states = states
        self.spaces = spaces

        self.matrix = {key: np.random.rand(self.states, *space.shape) for key, space in spaces.items()}
        self.matrix = {key: matrix / np.sum(matrix, axis=1)[:, np.newaxis] for key, matrix in self.matrix.items()}

    def p(self, state, value=None):
        if value is None:
            return {key: matrix[state] for key, matrix in self.matrix.items()}

    def sample(self, state):
        return {key: np.random.choice(space, p=matrix[state]) for
                key, matrix, space in zip(self.matrix.keys(), self.matrix.values(), self.spaces.values())}

    def index_single(self, key, x):
        return np.argmin(np.abs(self.spaces[key] - x))

    def index(self, x):
        return {key: np.argmin(np.abs(space - x[key])) for key, space in self.spaces.items()}

    def clip(self, x):
        return {key: space[index] for key, index, space in zip(self.spaces.keys(), self.index(x), self.spaces.values())}


class HMM:
    def __init__(self, states: int, space: np.ndarray):
        self.hidden_states = states
        self.observation_space = space
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
    # def mixture_sampling(self, T):
    #     return mixture_sampling_any(self.transition_matrix, self.emission_matrix, self.observation_space, T)


class MultivariateHMM:
    def __init__(self, states: int, spaces: Dict[str, np.ndarray]):
        self.hidden_states = states
        self.spaces = spaces

        self.transition_matrix = TransitionMatrix(states)
        self.emission_matrix = EmissionMatrix(states, spaces)

        self.current_state = np.random.choice(np.arange(self.hidden_states))

    def mixture_sampling(self, T):
        df = pd.DataFrame([], columns=list(self.spaces.keys()), index=np.arange(T))
        for t in range(T):
            self.current_state = self.transition_matrix.transition(self.current_state)
            observation = self.emission_matrix.sample(self.current_state)
            df.loc[t] = observation

        return df

    def forward(self, x: pd.DataFrame):
        T = len(x)

        alpha = {
            key: np.zeros((T, self.hidden_states)) for key in self.spaces.keys()
        }

        p = self.emission_matrix.p(self.current_state)
        k = self.emission_matrix.index(x.iloc[0])

        for key in self.spaces.keys():
            alpha[key][0] = p[key][k[key]]

        for t in range(1, T):
            for key in self.spaces.keys():
                a = 1
                alpha[key][t] = (
                        np.sum(alpha[key][t - 1] * self.transition_matrix.p(self.current_state))
                        * self.emission_matrix.p(self.current_state)[key][self.emission_matrix.index_single(key, x[key].iloc[t])]
                )
                alpha[key][t] /= np.sum(alpha[key][t])

        return alpha

    def backward(self, x: pd.DataFrame):
        T = len(x)

        beta = {
            key: np.zeros((T, self.hidden_states)) for key in self.spaces.keys()
        }

        for key in self.spaces.keys():
            beta[key][T - 1] = 1

        for t in range(T - 2, -1, -1):
            for key in self.spaces.keys():
                beta[key][t] = np.sum(
                    self.transition_matrix.p(self.current_state) *
                    self.emission_matrix.p(self.current_state)[key][self.emission_matrix.index_single(key, x[key].iloc[t + 1])] *
                    beta[key][t + 1],
                )
                beta[key][t] /= np.sum(beta[key][t])

        return beta

    def baum_welch(self, x: pd.DataFrame, iterations):
        T = len(x)

        for k in range(iterations):
            alpha = self.forward(x)
            beta = self.backward(x)

            gamma = {
                key: np.zeros((T, self.hidden_states)) for key in self.spaces.keys()
            }

            for key in self.spaces.keys():
                gamma[key] = np.multiply(alpha[key], beta[key])
                gamma[key] /= np.sum(gamma[key], axis=1)[:, np.newaxis]

            xi = {
                key: np.zeros((T - 1, self.hidden_states, self.hidden_states)) for key in self.spaces.keys()
            }

            for t in range(T - 1):
                for key in self.spaces.keys():
                    for i in range(self.hidden_states):
                        for j in range(self.hidden_states):
                            xi[key][t, i, j] = (
                                    alpha[key][t, i]
                                    * self.transition_matrix.p(i)[j]
                                    * self.emission_matrix.p(j)[key][self.emission_matrix.index_single(key, x[key].iloc[t + 1])]
                                    * beta[key][t + 1, j]
                            )
                    xi[key][t] /= np.sum(xi[key][t])

            for key in self.spaces.keys():
                self.transition_matrix.matrix = np.sum(xi[key], axis=0) / np.sum(gamma[key], axis=0)[:, np.newaxis]
                self.transition_matrix.matrix /= np.sum(self.transition_matrix.matrix, axis=1)[:, np.newaxis]
                # gamma has shape (T, hidden_states)
                # emission_matrix has shape (hidden_states, len(space))
                gamma_sum = np.sum(gamma[key], axis=0)
                for i in range(self.hidden_states):
                    for j in range(len(self.spaces[key])):
                        self.emission_matrix.matrix[key][i, j] = np.sum(gamma[key][:, i] * (x[key] == self.spaces[key][j])) / gamma_sum[i]

            if k % 10 == 0:
                print(k)


def main():
    def test_hmm():
        hidden_states = 3

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

        # data = mixture_sampling_normal(true_transition_matrix, true_emission_matrix, observation_space, 1000)
        data = []

        hmm = HMM(hidden_states, observation_space)

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

    vhmm = MultivariateHMM(2, {"a": np.linspace(-1, 1, 100), "b": np.linspace(-1, 1, 100)})
    x = vhmm.mixture_sampling(1000)

    vhmm.baum_welch(x, 100)

    print(vhmm.transition_matrix.matrix)
    print(vhmm.emission_matrix.matrix)


if __name__ == "__main__":
    main()
