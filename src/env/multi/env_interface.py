from urllib.parse import urlparse

import httpx
from dxlib.interfaces import Interface, Server, Protocols

from src.lib import Dealer


class EnvInterface(Interface):
    def __init__(self):
        self.router: Server | None = None
        self.snapshot_server: Server | None = None

    def register(self, server: Server, snapshot_server: Server):
        self.router = server
        self.snapshot_server = snapshot_server

    def register_service(self, endpoints):
        # [{'method': 'ROUTER', 'name': 'router', 'path': 'tcp://localhost:5001'}] -> parse "path"
        for endpoint in endpoints:
            if endpoint["method"] == "router":
                result = urlparse(endpoint["path"])
                self.router = Server(result.hostname, result.port, protocol=Protocols.TCP)
            elif endpoint["method"] == "snapshot":
                result = urlparse(endpoint["path"])
                self.snapshot_server = Server(result.hostname, result.port, protocol=Protocols.HTTP)

    def step(self, action, dealer: Dealer):
        response = dealer.send(action, self.router.url)
        return response

    def state(self):
        with httpx.Client() as client:
            response = client.get(self.snapshot_server.url)
            return response.json()
