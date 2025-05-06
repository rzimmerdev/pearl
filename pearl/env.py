import logging
import threading
from urllib.parse import urlparse

from httpx import ConnectError

from dxlib.network.servers import Server
from dxlib.network.interfaces.internal import MeshInterface
from dxlib.network.servers.http.fastapi import FastApiServer

from pearl.config import MeshConfig
from pearl.envs.multi import MarketEnvService


def main(host,
         mesh_config: MeshConfig,
         max_envs: int,
         env_id=None):
    server_intervals = range(5001, 5001 + max_envs)
    router_intervals = range(5002 + max_envs, 5002 + 2 * max_envs)

    mesh = MeshInterface()
    mesh.register(Server(mesh_config.host, mesh_config.port))
    services = mesh.search_services()

    if env_id is not None:
        server_port = server_intervals[env_id]
        router_port = router_intervals[env_id]
    else:
        server_intervals = set(server_intervals)
        router_intervals = set(router_intervals)
        for service in services:
            if service != "market_env":
                continue

            for instance_uuid, instance in services[service].items():
                endpoints = instance["endpoints"]

                for route, endpoint in endpoints.items():
                    for method, details in endpoint.items():
                        if method == "GET" or method == "POST":
                            # parse route == "http://localhost:5001/whatever"
                            parsed = urlparse(route)
                            server_intervals.discard(int(parsed.port))
                        elif method == "router":
                            parsed = urlparse(route)
                            router_intervals.discard(int(parsed.port))
        server_port = server_intervals.pop()
        router_port = router_intervals.pop()
    server = FastApiServer(host, server_port, log_level=logging.WARNING)
    env = MarketEnvService(host, router_port, n_levels=10, starting_value=100, dt=1 / 252 / 6.5 / 60)

    server.register(env)
    thread = threading.Thread(target=server.run)

    try:
        thread.start()
        env.start()
        mesh.register_service(env.data(server.url))
        env.router.use_mesh(mesh_config.name, mesh_config.host, mesh_config.port, env.name, env.service_id)
        while env.running:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        env.stop()
        server.stop()
        thread.join()
        try:
            mesh.deregister_service(env.name, env.service_id)
        except ConnectError:
            pass
