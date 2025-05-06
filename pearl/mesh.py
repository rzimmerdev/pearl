import logging

from dxlib.network.interfaces.internal import MeshService
from dxlib.network.servers.http.fastapi import FastApiServer

from pearl.config import MeshConfig


def main(mesh_config: MeshConfig):
    name, host, port = mesh_config
    mesh = MeshService(name)

    server = FastApiServer(host, port, log_level=logging.WARNING)
    server.register(mesh)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down")


if __name__ == "__main__":
    main(MeshConfig())
