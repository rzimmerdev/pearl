from dxlib.interfaces.internal import MeshService
from dxlib.interfaces.services.http.fastapi import FastApiServer

from pearl.load_mesh import load_config


def main(config=None):
    _, mesh_name, mesh_host, mesh_port = load_config(config)
    mesh = MeshService(mesh_name)

    server = FastApiServer(mesh_host, mesh_port)
    server.register(mesh)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down")


if __name__ == "__main__":
    main()
