import json
from typing import Callable, List, Any, Dict, Tuple
from uuid import uuid4
import zmq
from dxlib.interfaces import Protocols
from dxlib.interfaces.internal.mesh import MeshInterface
from dxlib.interfaces.services import Server


class Dealer:
    def __init__(self, routers=None, max_sockets=100):
        self.routers = routers or {}
        self.uuid = uuid4().hex.encode()
        self.context = zmq.Context()
        self.sockets: Dict[str, zmq.Socket] = {}
        self.max_sockets = max_sockets
        self.mesh = None
        self.mesh_name = None

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        self.mesh = MeshInterface()
        self.mesh_name = mesh_name
        self.mesh.register(Server(mesh_host, mesh_port))

    def register(self, routers):
        self.routers = routers

    def _get_socket(self, service_id: str):
        if service_id not in self.sockets:
            if len(self.sockets) >= self.max_sockets:
                oldest_service = next(iter(self.sockets))
                self.sockets[oldest_service].close()
                del self.sockets[oldest_service]

            socket = self.context.socket(zmq.DEALER)
            socket.setsockopt(zmq.IDENTITY, self.uuid)
            socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 seconds send timeout
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 seconds receive timeout
            socket.connect(self.routers[service_id])
            self.sockets[service_id] = socket
        return self.sockets[service_id]

    @staticmethod
    def _handle_response(frames):
        return frames

    def send(self, messages: Dict[str, bytes] | bytes, handler: Callable[[List[bytes]], Any] = None):
        handler = self._handle_response if handler is None else handler
        responses = {}

        if isinstance(messages, bytes):
            messages = {service_id: messages for service_id in self.routers.keys()}

        for service_id, message in messages.items():
            if service_id not in self.routers:
                print(f"Unknown service_id: {service_id}")
                continue

            socket = self._get_socket(service_id)
            try:
                socket.send_multipart([message])
            except zmq.Again:
                print(f"Timeout reached when sending to {self.routers[service_id]}")
                continue

            try:
                frames = socket.recv_multipart()
                responses[service_id] = handler(*[json.loads(frame) for frame in frames])
            except zmq.Again:
                print(f"Timeout reached when receiving from {self.routers[service_id]}")

        return responses



    def close(self):
        for socket in self.sockets.values():
            socket.close()
        self.sockets.clear()
        self.context.term()
