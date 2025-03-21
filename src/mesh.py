from dxlib.interfaces.internal import MeshService
from dxlib.interfaces.services.http.fastapi import FastApiServer


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
