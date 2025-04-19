import asyncio
import json
from typing import Callable, List, Any, Dict, Union
from uuid import uuid4
import zmq
import zmq.asyncio
from dxlib.network.interfaces.internal import MeshInterface
from dxlib.network.servers import Server


class Dealer:
    def __init__(self, routers=None, max_sockets=100):
        self.routers = routers or {}
        self.uuid = uuid4().hex.encode()
        self.context = zmq.asyncio.Context()
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

    def _get_socket(self, service_id: str) -> zmq.Socket:
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
        return frames[0]

    async def _send_multipart(self, service_id, message):
        socket = self._get_socket(service_id)
        try:
            await socket.send_multipart([message])  # Non-blocking async send
        except zmq.Again:
            print(f"Timeout reached when sending to {self.routers[service_id]}")
            return service_id, None
        try:
            # noinspection PyUnresolvedReferences
            frames = await socket.recv_multipart()  # Non-blocking async recv
            return service_id, [json.loads(frame) for frame in frames]
        except zmq.Again:
            print(f"Timeout reached when receiving from {self.routers[service_id]}")
            return service_id, None

    def send(self, messages: Union[Dict[str, bytes], bytes], handler: Callable[[List[bytes]], Any] = None):
        handler = self._handle_response if handler is None else handler

        if isinstance(messages, bytes):
            messages = {service_id: messages for service_id in self.routers.keys()}

        async def _send_all():
            tasks = [self._send_multipart(service_id, message) for service_id, message in messages.items() if service_id in self.routers]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_send_all())

        responses = {service_id: handler(result) for service_id, result in results if result is not None}
        return responses

    def close(self):
        for socket in self.sockets.values():
            socket.close()
        self.sockets.clear()
        self.context.term()
