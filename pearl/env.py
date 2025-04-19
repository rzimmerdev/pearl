import logging
import threading
from urllib.parse import urlparse

from httpx import ConnectError

from dxlib.network.servers import Server
from dxlib.network.interfaces.internal import MeshInterface
from dxlib.network.servers.http.fastapi import FastApiServer

from pearl.envs.multi import MarketEnvService
from pearl.load_mesh import load_config


def main(max_envs: int, config=None):
    host, mesh_name, mesh_host, mesh_port = load_config(config)
    server_intervals = set(range(5001, 5001 + max_envs))
    router_intervals = set(range(5002 + max_envs, 5002 + 2*max_envs))

    mesh = MeshInterface()
    mesh.register(Server(mesh_host, mesh_port))
    # get existing services
    services = mesh.search_services()

    # iterate over services endpoints to get the host and port of the existing envs and decide the next port
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

    server = FastApiServer(host, server_intervals.pop(), log_level=logging.WARNING)
    env = MarketEnvService(host, router_intervals.pop(), n_levels=10, starting_value=100, dt=1 / 252 / 6.5 / 60)

    server.register(env)
    thread = threading.Thread(target=server.run)

    try:
        thread.start()
        env.start()
        mesh.register_service(env.data(server.url))
        env.router.use_mesh(mesh_name, mesh_host, mesh_port, env.name, env.service_id)
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
