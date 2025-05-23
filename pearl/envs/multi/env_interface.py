import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import httpx
from urllib.parse import urlparse

from dxlib.network.interfaces import Interface
from dxlib.network.servers import Server, Protocols
from dxlib.network.interfaces.internal import MeshInterface
from httpx import ConnectError

from pearl.lib import Dealer


@dataclass
class Env:
    router: Server
    snapshot: Server
    agent_id: Optional[str] = None


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
        keys = list(response.keys())
        user_id = None
        for env_id, msg in response.items():
            if msg.get('status', None) != 'connected':
                raise ValueError(f"Failed to connect to env: {env_id}")
            self._envs[env_id].agent_id = msg['user']
            user_id = msg['user']
        print(f"Registered user with id {user_id} to {len(keys)} environments.")

    def step(self, actions, timesteps):
        response = self._send({
            env_id: self.encode({"type": "action", "data": action, "timestep": timesteps[env_id]})
            for env_id, action in actions.items()})
        return response

    def reset(self, env_ids: List[str]):
        if not env_ids:
            return
        response = self._send({
            env_id: self.encode({"type": "reset"}) for env_id in env_ids
        })
        return response

    def state(self) -> Dict[str, dict]:
        with httpx.Client() as client:
            states = {}
            for env_id, env in self._envs.items():
                url = env.snapshot.url
                response = client.get(f"{url}/state", params={"agent_id": env.agent_id})
                states[env_id] = response.json()
            return states

    def snapshot(self, env_id):
        with httpx.Client() as client:
            env = self._envs[env_id]
            url = env.snapshot.url
            response = client.get(f"{url}/snapshot", params={"agent_id": env.agent_id})
            response.raise_for_status()
            return response.json()

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
