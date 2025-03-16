import asyncio
from collections import defaultdict

from src.env.lib import Router
from src.env.lib.timer import Timer
from src.env.multi.multi_env import MarketEnv


class MarketEnvService(MarketEnv):
    def __init__(self, host, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = Router(host, port)
        self.action_buffer = defaultdict()
        self.timer = Timer(1.0)
        self.connections = set()

        self.waiting = True
        self.response = {}

    def handle(self, uuid, msg):
        # decide based on msg:
        # {"type": "action", "data": [0.1, 0.2, 0.3, 0.4]}
        # {"type": "close"}
        # {"type": "connect"}
        if not isinstance(msg, dict):
            print(f"Invalid message received: {msg}")
            return "Invalid message received"
        if msg["type"] == "close":
            self.connections.remove(uuid)
            return
        elif msg["type"] == "connect":
            self.connections.add(uuid)
            self.add_user(uuid)
            return

        if msg["type"] != "action" or "data" not in msg:
            return

        if uuid not in self.connections:
            return "User not connected"

        self.action_buffer[uuid] = msg["data"]
        self.waiting = True

        if len(self.action_buffer) == 1:
            self.timer.start(self.execute_step, True, self.action_buffer)

        if len(self.action_buffer) == len(self.connections):
            self.timer.stop()
            self.execute_step(self.action_buffer)

        while self.waiting:
            pass

        response = self.response

        return {
            "state": response["state"][uuid].tolist(),
            "reward": response["reward"][uuid],
            "done": response["done"],
            "trunc": response["trunc"],
        }

    def execute_step(self, action_buffer):
        self.waiting = False
        actions = {uuid: action_buffer[uuid] for uuid in action_buffer}

        state, reward, done, trunc, _ = self.step(actions)
        self.response = {"state": state, "reward": reward, "done": done, "trunc": trunc}

    def start(self, mesh_name=None, mesh_host="localhost", mesh_port=8000):
        """Run router async and timer async"""
        self.router.bind()

        if mesh_name is not None:
            self.router.use_mesh(mesh_name, mesh_host, mesh_port)

        self.router.start(self.handle)

    def stop(self):
        self.timer.stop()


if __name__ == "__main__":
    env = MarketEnvService("localhost", 5000, n_levels=10, starting_value=100)

    try:
        env.start("pearl")
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
