import json

from src.env.lib import Dealer


class MultiAgent(Dealer):
    def __init__(self):
        super().__init__()

    def send(self, content, *args, **kwargs):
        msg = json.dumps(content).encode()
        response = super().send(msg)
        return response

if __name__ == "__main__":
    agent = MultiAgent()
    agent.use_mesh("pearl", "localhost", 8000)
    connect = {"type": "connect"}
    agent.send(connect)
    # action = {"type": "action", "data": [0.1, 0.2, 0.3, 0.4]}
    # print(agent.send(action))

    response = None
    for _ in range(20):
        action = {"type": "action", "data": [0.1, 0.2, 0.3, 0.4]}
        response = agent.send(action)

    print(response)
