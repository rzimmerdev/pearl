import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import httpx
from urllib.parse import urlparse

from dxlib.interfaces import Interface, Server, Protocols, ServiceData
from dxlib.interfaces.internal import MeshInterface
from httpx import ConnectError

from src.lib import Dealer


@dataclass
class Env:
    router: Server
    snapshot: Server
    agent_id: str | None = None


class EnvInterface(Interface):
    def __init__(self):
        self.mesh = MeshInterface()
        self.dealer = Dealer()
        self._envs: Dict[str, Env] = {}

    def _load_services(self, endpoints) -> Tuple[Server, Server]:
        # [{'method': 'ROUTER', 'name': 'router', 'path': 'tcp://localhost:5001'}] -> parse "path"
        router = None
        snaphshot = None
        for route in endpoints:
            for method in endpoints[route]:
                if method == "router":
                    result = urlparse(route)
                    router = Server(result.hostname, result.port, protocol=Protocols.TCP)
                elif method == "GET" and endpoints[route][method]['handler'] == 'get_state':
                    result = urlparse(route)
                    snaphshot = Server(result.hostname, result.port, protocol=Protocols.HTTP)
        return router, snaphshot

    def encode(self, data):
        # encode dict to bytes
        # return bytes
        return json.dumps(data).encode()

    def _send(self, data):
        return self.dealer.send(data)

    def register_user(self):
        response = self._send(self.encode({"type": "connect"}))
        print(f"Registered user with agent_id: {response}")
        for env_id, msg in response.items():
            if msg.get('status', None) != 'connected':
                raise ValueError(f"Failed to connect to env: {env_id}")
            self._envs[env_id].agent_id = msg['user']

    def step(self, actions, timesteps):
        response = self._send({
            env_id: self.encode({"type": "action", "data": action, "timestep": timesteps[env_id]})
            for env_id, action in actions.items()})
        return response

    def state(self) -> Dict[str, dict]:
        with httpx.Client() as client:
            states = {}
            for env_id, env in self._envs.items():
                url = env.snapshot.url
                response = client.get(f"{url}/state", params={"agent_id": env.agent_id})
                states[env_id] = response.json()
            return states

    def reset(self, env_ids: List[str]):
        if not env_ids:
            return
        response = self._send({
            env_id: self.encode({"type": "reset"}) for env_id in env_ids
        })
        return response

    def __iter__(self):
        return iter(self._envs.keys())

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        self.mesh.register(Server(mesh_host, mesh_port))
        try:
            instances = self.mesh.get_service("market_env")
        except httpx.ConnectError as e:
            raise ConnectError(f"Could not connect to mesh service: {e}")
        if len(instances) == 0:
            raise ValueError(f"No instances found for mesh service '{mesh_name}. Unable to use remote environment.")
        for service in instances:
            endpoints = service.endpoints
            router, server = self._load_services(endpoints)
            self._envs[service.service_id] = Env(router, server)

        self.dealer.register({service_id: env.router.url for service_id, env in self._envs.items()})


if __name__ == "__main__":
    mesh_name = os.getenv("MESH_NAME")
    mesh_host = os.getenv("MESH_HOST")
    mesh_port = int(os.getenv("MESH_PORT"))
    envs = EnvInterface()
    envs.use_mesh(mesh_name, mesh_host, mesh_port)
    envs.register_user()
    print(envs.state())

    response = envs.step([0.1, 0.2, 0.3, 0.4])
    print(response)
