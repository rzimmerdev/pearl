import json
import threading
from uuid import uuid4
import logging
from asyncio import CancelledError
from json import JSONDecodeError

import zmq
import zmq.asyncio
import asyncio

from dxlib.interfaces import Protocols
from dxlib.interfaces.internal.mesh import MeshInterface
from dxlib.interfaces.services import ServiceData, Server
from httpx import HTTPStatusError

log_format = "%(levelname)s:     %(message)s"


class Router:
    def __init__(self, host, port, level=logging.INFO):
        self.context = None
        self.socket = None
        self.host = host
        self.port = port
        self.mesh: MeshInterface | None = None

        self.service = None

        self._running = asyncio.Event()
        self.listen_task = None
        self.task = None
        self.thread = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(handler)

    def bind(self):
        self.socket.bind(f"tcp://{self.host}:{self.port}")

    def unbind(self):
        if self.socket:
            self.socket.unbind(f"tcp://{self.host}:{self.port}")

    def use_mesh(self,
                 mesh_name,
                 mesh_host,
                 mesh_port,
                 service_name: str = "router",
                 service_id: str = uuid4().hex
                 ):
        self.mesh = MeshInterface()
        self.mesh.register(Server(mesh_host, mesh_port))
        route = f"tcp://{self.host}:{self.port}"
        method = Protocols.ROUTER.value
        self.service = ServiceData(
            name=service_name,
            service_id=service_id,
            endpoints={
                route: {method:{"path": f"tcp://{self.host}:{self.port}", "method": "ROUTER", "name": "router"}}
            },
            tags={"router"},
        )
        try:
            self.mesh.register_service(self.service)
        except HTTPStatusError as e:
            self.logger.error(e.response.text)
            return
        self.logger.info(f"Using mesh '{mesh_name}' at http://{mesh_host}:{mesh_port}")
        self.logger.info(f"Registered service {self.service.name} with id {self.service.service_id}")

    def remove_mesh(self):
        if self.mesh and self.service:
            self.logger.info(f"Deregistering service {self.service.name} with id {self.service.service_id}")
            self.mesh.deregister_service(self.service.name, self.service.service_id)

    async def listen(self, handler):
        while self.running:
            try:
                identity, message = await self.socket.recv_multipart()
            except zmq.error.Again:
                continue
            except zmq.error.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                raise
            except CancelledError:
                break

            try:
                content = json.loads(message)
                response = handler(identity, content)
                await self.socket.send_multipart([identity, json.dumps(response).encode()])
            except JSONDecodeError as e:
                print(f"Invalid message received: {message}")
                await self.socket.send_multipart([identity, json.dumps({"error": "Expected JSON message"}).encode()])
            except Exception as e:
                await self.socket.send_multipart([identity, json.dumps({"error": "Error processing message: " + str(e)}).encode()])
                raise

    @property
    def running(self):
        return self._running.is_set()

    def open(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def close(self):
        if self.socket:
            self.socket.close()
        if self.task:
            self.task.cancel()
        if self.context:
            self.context.term()

    def start(self, handler):
        self._running.set()
        self.open()
        self.bind()
        self.thread = threading.Thread(target=asyncio.run, args=(self.listen(handler),))
        self.thread.start()

        self.logger.info(f"Started server process {self.thread.ident}")
        self.logger.info("Waiting for application startup.")
        self.logger.info("Application startup complete.")
        self.logger.info(f"Router running on tcp://{self.host}:{self.port} (Press CTRL+C to quit)")

    def stop(self):
        self._running.clear()
        self.unbind()
        self.close()
        if self.thread:
            self.thread.join()

if __name__ == "__main__":
    router = Router("localhost", 5000)
    router.use_mesh("pearl", "localhost", 8000)

    try:
        router.start(print)
        while router.running:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        router.stop()
        router.remove_mesh()
