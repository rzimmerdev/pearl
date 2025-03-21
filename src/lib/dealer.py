import json
from typing import Callable, List, Any
from urllib.parse import urlparse
from uuid import uuid4

import zmq
from dxlib.interfaces import Protocols

from dxlib.interfaces.internal.mesh import MeshInterface
from dxlib.interfaces.services import Server


class Dealer:
    def __init__(self, routers=None):
        self.routers = routers or {}
        self.sockets = {}
        self.mesh = None
        self.mesh_name = None
        self.uuid = uuid4().hex.encode()

    def use_mesh(self, mesh_name, mesh_host, mesh_port):
        self.mesh = MeshInterface()
        self.mesh_name = mesh_name
        self.mesh.register(Server(mesh_host, mesh_port))

    def register(self, routers):
        self.routers = routers
        self.sockets = {service_id: self._create_socket() for service_id in self.routers.keys()}

    def _create_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, self.uuid)
        socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 seconds send timeout
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 seconds receive timeout
        return context, socket

    @staticmethod
    def _handle_response(frames):
        return frames

    def send(self, messages: dict | bytes, handler: Callable[[List[bytes]], Any] = None):
        handler = self._handle_response if handler is None else handler

        responses = {}

        if isinstance(messages, bytes):
            messages = {service_id: messages for service_id in self.routers.keys()}

        for service_id, message in messages.items():
            context, socket = self._create_socket()
            route = self.routers[service_id]
            socket.connect(route)

            try:
                socket.send_multipart([message])
            except zmq.Again:
                print(f"Timeout reached when sending to {route}")
            try:
                frames = socket.recv_multipart()
                responses[service_id] = handler(*[json.loads(frame) for frame in frames])
            except zmq.Again:
                print(f"Timeout reached when receiving from {route}")
            finally:
                socket.disconnect(route)
                socket.close()
                context.term()

        return responses
