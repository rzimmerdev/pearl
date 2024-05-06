from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


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
    def __init__(self, states: int, spaces: Dict[str, np.ndarray]):
        self.hidden_states = states
        self.spaces = spaces

        self.transition_matrix = TransitionMatrix(states)
        self.emission_matrix = EmissionMatrix(states, spaces)

        self.current_state = np.random.choice(np.arange(self.hidden_states))

    def mixture_sampling(self, T) -> pd.DataFrame:  # shape (T, len(spaces))
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

        d = iterations // 10

        for k in tqdm(range(iterations)):
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


def main():
    hmm = HMM(2, {"a": np.linspace(-1, 1, 100), "b": np.linspace(-1, 1, 100)})
    x = hmm.mixture_sampling(1000)

    hmm.baum_welch(x, 100)

    print(hmm.transition_matrix.matrix)
    print(hmm.emission_matrix.matrix)


if __name__ == "__main__":
    main()
