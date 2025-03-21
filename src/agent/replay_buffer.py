
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, log_prob, reward, done, trunc=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self):
        return zip(self.states, self.actions, self.log_probs, self.rewards, self.dones)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []