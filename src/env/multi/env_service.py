import threading
from collections import defaultdict
from uuid import uuid4

from dxlib.interfaces.internal import MeshInterface

from src.lib import Router
from src.lib.timer import Timer
from src.env.multi.multi_env import MarketEnv

from dxlib.interfaces import HttpEndpoint, Service, Server, ServiceModel
from dxlib.interfaces.services.http.fastapi import FastApiServer


class MarketEnvService(Service, MarketEnv):
    def __init__(self,
                 host,
                 port,
                 *args,
                 **kwargs
                 ):
        Service.__init__(self, "market_env", uuid4().hex)
        MarketEnv.__init__(self, *args, **kwargs)
        self.router = Router(host, port)
        self.action_buffer = defaultdict()
        self.timer = Timer(1.0)
        self.connections = set()

        self.waiting = True
        self.response = {}
        self.mesh = None

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

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        model = self.to_model()
        self.router.use_mesh(mesh_name, mesh_host, mesh_port, model.name, model.service_id)
        self.mesh = MeshInterface()
        self.mesh.register(Server(mesh_host, mesh_port))
        self.mesh.register_service(model)

    def deregister_mesh(self, mesh_name, mesh_host, mesh_port):
        self.router.remove_mesh()
        self.mesh = MeshInterface()
        self.mesh.register(Server(mesh_host, mesh_port))
        self.mesh.deregister_service(self.name, self.service_id)

    def start(self, mesh_name=None, mesh_host="localhost", mesh_port=8000):
        """Run router async and timer async"""
        # start FastApiServer.Server
        if mesh_name is not None:
            self.use_mesh(mesh_name, mesh_host, mesh_port)

        self.router.start(self.handle)

    def stop(self):
        self.timer.stop()
        self.router.stop()
        if self.mesh:
            self.deregister_mesh("pearl", "localhost", 8000)

    @property
    def running(self):
        return self.router.running

    @HttpEndpoint.post("/state")
    def get_state(self, agent_id):
        return self.state(agent_id)


if __name__ == "__main__":
    host = "localhost"
    server = FastApiServer(host, 5000)
    env = MarketEnvService(host, 5001, n_levels=10, starting_value=100)

    server.register(env)
    thread = threading.Thread(target=server.run)
    try:
        thread.start()
        env.start("pearl")
        while env.running:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
        server.stop()
        thread.join()
