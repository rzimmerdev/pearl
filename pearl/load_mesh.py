import os


def load_config(config=None):
    config = config or {}
    host = config.get("HOST", os.getenv("HOST", "localhost"))
    mesh_name = config.get("MESH_NAME", os.getenv("MESH_NAME", "pearl"))
    mesh_host = config.get("MESH_HOST", os.getenv("MESH_HOST", "localhost"))
    mesh_port = int(config.get("MESH_PORT", os.getenv("MESH_PORT", 5000)))

    return host, mesh_name, mesh_host, mesh_port
