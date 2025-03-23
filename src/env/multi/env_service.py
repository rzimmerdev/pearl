import asyncio
import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Type
from uuid import uuid4

from dxlib.storage import T
from httpx import HTTPStatusError, ConnectError

from src.lib import Router
from src.lib.timer import Timer
from src.env.multi.multi_env import MarketEnv

from dxlib.interfaces import HttpEndpoint, Service, Server
from dxlib.interfaces.internal import MeshInterface
from dxlib.interfaces.services.http.fastapi import FastApiServer


@dataclass
class AgentModel:
    agent_id: str


class MarketEnvService(Service, MarketEnv):
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        pass

    def to_dict(self) -> dict:
        pass

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

        self.sync = asyncio.Condition()
        self.execute_lock = asyncio.Event()

    async def handle(self, identity, content):
        """Handle different types of messages asynchronously."""
        uuid = identity.decode()

        if not isinstance(content, dict):
            print(f"Invalid message received: {content}")
            return {"error": "Invalid message received"}

        if content["type"] == "close":
            self.connections.remove(uuid)
            return {"status": "closed"}

        elif content["type"] == "connect":
            self.connections.add(uuid)
            self.add_user(uuid)
            self.router.logger.info(f"User connected: {uuid}")
            return {"status": "connected", "user": uuid}

        if content["type"] != "action" or "data" not in content:
            return {"error": "Invalid action message format"}

        if uuid not in self.connections:
            return {"error": "User not connected"}

        # Store the action data for the user
        if self.execute_lock.is_set():
            return {"error": "Environment is busy, try again later",
                    "timeout": True}
        elif "timestep" not in content:
            return {"error": "No desired timestep in message", "timeout": True}
        elif content.get("timestep", 0) < self.timestep:
            return {"error": "Old timestep, send message before market timeout in 'timestep' (s) field",
                    "timeout": True}

        async with self.sync:
            self.action_buffer[uuid] = content["data"]
            if len(self.action_buffer) == len(self.connections):
                self.timer.stop()
                await self.execute_step()
            elif len(self.action_buffer) == 1:
                self.timer.start(self.execute_step, True, self.action_buffer)
                await self.sync.wait()
            else:
                await self.sync.wait()
        return {
            "state": self.response["state"][uuid].tolist(),
            "reward": self.response["reward"][uuid],
            "done": self.response["done"],
            "trunc": self.response["trunc"],
            "details": {"timestep": self.timestep}
        }

    async def execute_step(self):
        self.execute_lock.set()
        actions = {uuid: self.action_buffer[uuid] for uuid in self.action_buffer}
        self.action_buffer.clear()

        state, reward, done, trunc, _ = self.step(actions)

        self.response = {"state": state, "reward": reward, "done": done, "trunc": trunc}
        self.sync.notify_all()
        self.execute_lock.clear()

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        service_data = self.data()
        self.router.use_mesh(mesh_name, mesh_host, mesh_port, service_data.name, service_data.service_id)
        self.mesh = MeshInterface()
        self.mesh.register(Server(mesh_host, mesh_port))
        try:
            self.mesh.register_service(service_data)
        except HTTPStatusError as e:
            response = json.loads(e.response.text)
            self.router.logger.error("Error registering service: %s", json.dumps(response, indent=2))
            return

    def deregister_mesh(self):
        self.router.remove_mesh()

    def start(self):
        """Run router async and timer async"""
        # start FastApiServer.Server
        self.execute_lock.clear()
        self.router.start(self.handle)

    def stop(self):
        self.timer.stop()
        self.router.stop()

    @property
    def running(self):
        return self.router.running

    @HttpEndpoint.get("/state")
    def get_state(self, agent_id):
        if not self.verify_id(agent_id):
            raise ValueError(f"Invalid agent_id {agent_id} for environment.")
        return self.state(agent_id).tolist()

    @HttpEndpoint.post("/agent")
    def register_user(self, agent: AgentModel):
        agent_id = agent.agent_id
        if self.verify_id(agent_id):
            pass
        self.add_user(agent_id)


if __name__ == "__main__":
    host = "localhost"
    mesh_host = "localhost"
    server = FastApiServer(host, 5000)
    env = MarketEnvService(host, 5001, n_levels=10, starting_value=100, dt=1 / 252 / 6.5 / 60)
    mesh = MeshInterface()
    mesh.register(Server(mesh_host, 8000))

    server.register(env)
    thread = threading.Thread(target=server.run)

    try:
        thread.start()
        env.start()
        mesh.register_service(env.data(server.url))
        env.router.use_mesh("pearl", mesh_host, 8000, env.name, env.service_id)
        while env.running:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
        server.stop()
        thread.join()
        try:
            mesh.deregister_service(env.name, env.service_id)
        except ConnectError:
            pass
