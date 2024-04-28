import numpy as np


def mse(x, y):
    return np.mean((x - y) ** 2)


def mixture_sampling_normal(transition_matrix, emission_matrix, space, T):
    current_state = 0
    observations = []

    for t in range(T):
        current_state = np.random.choice(np.arange(len(transition_matrix)), p=transition_matrix[current_state])
        observation = np.random.choice(space, p=emission_matrix[current_state])

        observations.append(observation)

    return np.array(observations)


def mixture_sampling_any(transition_matrix, emission_distribution, space, T):
    current_state = 0
    observations = []

    for t in range(T):
        current_state = np.random.choice(np.arange(len(transition_matrix)), p=transition_matrix[current_state])
        histogram = emission_distribution[current_state]
        observation = np.random.choice(space, p=histogram)

        observations.append(observation)

    return np.array(observations)
