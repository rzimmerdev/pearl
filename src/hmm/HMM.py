import numpy as np
from scipy.stats import norm


class HMM:
    def __init__(self, hidden_states, observable_states, d):
        self.hidden_states = hidden_states
        self.observable_states = observable_states
        self.d = d
        self.transition_matrix = np.random.rand(hidden_states, hidden_states)
        self.emission_matrix = np.random.rand(hidden_states, observable_states)

        self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:, np.newaxis]
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1)[:, np.newaxis]


def forward(hmm, observations):
    alpha = np.zeros((len(observations), hmm.hidden_states))

    for i in range(hmm.hidden_states):
        alpha[0, i] = 1 / hmm.hidden_states * hmm.emission_matrix[i, observations[0]]

    for t in range(1, len(observations)):
        for i in range(hmm.hidden_states):
            alpha[t, i] = np.sum(
                [alpha[t - 1, j] * hmm.transition_matrix[j, i] * hmm.emission_matrix[i, observations[t]] for j in
                 range(hmm.hidden_states)])

    return alpha


def backward(hmm, observations):
    beta = np.zeros((len(observations), hmm.hidden_states))

    for i in range(hmm.hidden_states):
        beta[len(observations) - 1, i] = 1

    for t in range(len(observations) - 2, -1, -1):
        for i in range(hmm.hidden_states):
            beta[t, i] = np.sum(
                [beta[t + 1, j] * hmm.transition_matrix[i, j] * hmm.emission_matrix[j, observations[t + 1]] for j in
                 range(hmm.hidden_states)])

    return beta


def baum_welch(hmm, observations, iterations):
    for _ in range(iterations):
        # E-step
        alpha = forward(hmm, observations)
        beta = backward(hmm, observations)
        gamma = np.zeros((len(observations), hmm.hidden_states))
        xi = np.zeros((len(observations), hmm.hidden_states, hmm.hidden_states))

        for t in range(len(observations)):
            for i in range(hmm.hidden_states):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                gamma[t, i] /= np.sum(gamma[t])

                if t < len(observations) - 1:
                    for j in range(hmm.hidden_states):
                        xi[t, i, j] = alpha[t, i] * hmm.transition_matrix[i, j] * hmm.emission_matrix[
                            j, observations[t + 1]] * beta[t + 1, j]
                    xi[t] /= np.sum(xi[t])

        # M-step
        for i in range(hmm.hidden_states):
            for j in range(hmm.hidden_states):
                hmm.transition_matrix[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:, i])

        for j in range(hmm.hidden_states):
            for k in range(hmm.observable_states):
                hmm.emission_matrix[j, k] = np.sum(
                    [gamma[t, j] for t in range(len(observations)) if observations[t] == k]) / np.sum(gamma[:, j])

    return hmm


# Example usage
# generating data
hidden_normal_1 = (0, 1)
hidden_normal_2 = (5, 1)

true_transition = np.array([[0.7, 0.3], [0.3, 0.7]])
true_emission = np.array([[0.9, 0.1], [0.2, 0.8]])


def generate_data(n, hidden_normal_1, hidden_normal_2, true_transition, true_emission):
    data = []
    hidden_states = []
    hidden_state = 0

    for _ in range(n):
        hidden_state = 0 if hidden_state == 0 else 1
        hidden_states.append(hidden_state)
        data.append(np.random.normal(*hidden_normal_1) if hidden_state == 0 else np.random.normal(*hidden_normal_2))

    observations = [0 if hidden_state == 0 else 1 for hidden_state in hidden_states]

    return data, observations

data, observations = generate_data(100, hidden_normal_1, hidden_normal_2, true_transition, true_emission)

hmm = baum_welch(HMM(2, 2, 1), observations, 100)

print(hmm.transition_matrix)
print(true_transition)