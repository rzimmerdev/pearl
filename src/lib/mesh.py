from dxlib.interfaces.internal import MeshService
from dxlib.interfaces.services.http.fastapi import FastApiServer


class MeshProtocol(MeshService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def connection(self):
        return


if __name__ == "__main__":
    mesh = MeshService("pearl")

    server = FastApiServer("localhost", 8000)
    server.register(mesh)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Server stopped")
