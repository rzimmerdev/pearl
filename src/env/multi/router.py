import uuid
import zmq
import zmq.asyncio
import asyncio
import signal
from dxlib.interfaces.internal.mesh import MeshInterface
from dxlib.interfaces.services import ServiceModel, Server

class Router:
    def __init__(self, host, port):
        self.context = None
        self.socket = None
        self.open()
        self.host = host
        self.port = port
        self.mesh = None
        self.service = None
        self.running = asyncio.Event()
        self.listen_task = None

    def bind(self):
        self.socket.bind(f"tcp://{self.host}:{self.port}")

    def unbind(self):
        self.socket.unbind(f"tcp://{self.host}:{self.port}")

    def use_mesh(self, name, host, port):
        self.mesh = MeshInterface()
        self.mesh.register(Server(host, port))
        self.service = ServiceModel(
            name=name,
            service_id=uuid.uuid4().hex,
            endpoints=[{"path": f"tcp://{self.host}:{self.port}", "method": "ROUTER", "name": "router"}],
            tags=["router"],
        )
        self.mesh.register_service(self.service)
        print(f"Registered with mesh the id {self.service.service_id}")

    def remove_mesh(self):
        if self.mesh and self.service:
            self.mesh.deregister_service(self.service.name, self.service.service_id)

    async def listen(self):
        try:
            while self.running.is_set():
                try:
                    identity, message = await self.socket.recv_multipart()
                    print(f"Received from {identity}: {message.decode('utf-8')}")
                    await self.socket.send_multipart([identity, message])
                except zmq.error.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        break
                    else:
                        raise
        except asyncio.CancelledError:
            pass

    async def run(self):
        self.listen_task = asyncio.create_task(self.listen())
        await self.listen_task

    async def stop(self):
        self.running.clear()
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass

    def open(self):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

    def close(self):
        self.socket.close()
        self.context.term()

    def start(self):
        """Starts the router and manages lifecycle, including signal handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def shutdown():
            print("Received exit signal. Shutting down...")
            await self.stop()
            loop.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

        self.running.set()
        try:
            loop.run_until_complete(self.run())
        finally:
            pending_tasks = asyncio.all_tasks(loop)
            for task in pending_tasks:
                task.cancel()
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
            loop.close()


if __name__ == "__main__":
    router = Router("localhost", 5000)
    router.bind()
    router.use_mesh("pearl", "localhost", 8000)

    try:
        router.start()
    except KeyboardInterrupt:
        pass
    finally:
        router.remove_mesh()
        router.close()
