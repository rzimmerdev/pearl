from typing import Callable, List, Any

import zmq

from dxlib.interfaces.internal.mesh import MeshInterface
from dxlib.interfaces.services import Server


class Dealer:
    def __init__(self, routers=None):
        self.routers = routers or {}
        self.sockets = {}
        self.mesh = None
        self.mesh_name = None

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        self.mesh = MeshInterface()
        self.mesh.register(Server(mesh_host, mesh_port))
        self.mesh_name = mesh_name
        self.routers = {
            service["service_id"]: service["endpoints"] for service in
            self.mesh.get_service(self.mesh_name)
        }
        self.sockets = {
            service_id: self._create_socket() for service_id in self.routers.keys()
        }

    def _create_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        return context, socket

    @staticmethod
    def _handle_response(frames):
        return frames

    def send(self, message, handler: Callable[[List[bytes]], Any] = None):
        handler = self._handle_response if handler is None else handler

        responses = {}

        for service_id, endpoints in self.routers.items():
            socket = self.sockets[service_id][1]
            if len(endpoints) < 1:
                continue

            # get endpoint that has "name" == "router" [{...}, ..., {"name": "router", "path": "tcp://localhost:8000"} <- addr]
            addr = next(endpoint["path"] for endpoint in endpoints if endpoint["name"] == "router")
            socket.connect(addr)

            socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 seconds send timeout
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 seconds receive timeout

            try:
                socket.send_multipart([message])
                responses[service_id] = handler(socket.recv_multipart())

            except zmq.Again:
                print(f"Timeout reached when sending to {addr}")

            # disconnect but keep the socket
            socket.disconnect(addr)

        return responses

if __name__ == "__main__":
    dealer = Dealer()
    dealer.use_mesh("pearl", "localhost", 8000)
    response = dealer.send(b"Hello, World!")
    print(response)
