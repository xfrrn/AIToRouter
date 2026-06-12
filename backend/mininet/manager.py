# backend/mininet/manager.py
"""Manage Mininet Docker container lifecycle."""

from __future__ import annotations

import uuid
import time
import json
import os
import tempfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound, DockerException

from schemas.models import TopologyJSON
from mininet.templates import generate_mininet_script, build_nx_graph

MININET_IMAGE = "mnknowles/mininet:latest"


def check_docker_available() -> bool:
    """Quick check if Docker is reachable — does NOT pull any images."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class MininetManager:
    def __init__(self):
        self.client = docker.from_env()

    def _ensure_image(self):
        """Pull Mininet image if not present. Called lazily on first deploy."""
        try:
            self.client.images.get(MININET_IMAGE)
        except ImageNotFound:
            print(f"Pulling Mininet image {MININET_IMAGE}...")
            self.client.images.pull(MININET_IMAGE)
            print(f"Image {MININET_IMAGE} pulled successfully.")

    def deploy(self, topology: TopologyJSON, flows: list[dict]) -> tuple[str, str, str]:
        """Deploy topology to a Mininet container. Returns (container_id, exec_id, tmpdir)."""
        self._ensure_image()  # pull on first use, not at startup

        container_name = f"mininet-{uuid.uuid4().hex[:8]}"
        script = generate_mininet_script(topology)

        # Write script and flows to temp directory
        tmpdir = tempfile.mkdtemp(prefix="mininet-")
        script_path = os.path.join(tmpdir, "topo.py")
        flows_path = os.path.join(tmpdir, "flows.json")

        with open(script_path, "w") as f:
            f.write(script)
        with open(flows_path, "w") as f:
            json.dump(flows, f)

        container = self.client.containers.run(
            MININET_IMAGE,
            name=container_name,
            command="tail -f /dev/null",  # keep alive
            volumes={
                tmpdir: {"bind": "/tmp/topo", "mode": "rw"},
            },
            environment={"FLOWS_FILE": "/tmp/topo/flows.json"},
            privileged=True,  # Mininet needs network privileges
            detach=True,
            remove=False,
        )

        # Execute the topology script in background
        exec_result = self.client.api.exec_create(
            container.id,
            "python /tmp/topo/topo.py",
            stdout=True,
            stderr=True,
        )

        return container.id, exec_result["Id"], tmpdir

    def get_container(self, container_id: str):
        """Get a container by ID."""
        try:
            return self.client.containers.get(container_id)
        except NotFound:
            return None

    def exec_command(self, container_id: str, cmd: str) -> tuple[str, str]:
        """Execute a command in the container, return (stdout, stderr)."""
        container = self.get_container(container_id)
        if not container:
            return "", "Container not found"
        exit_code, output = container.exec_run(cmd, stdout=True, stderr=True)
        return output.decode("utf-8", errors="replace"), ""

    def stop_and_remove(self, container_id: str):
        """Stop and remove a container."""
        container = self.get_container(container_id)
        if container:
            container.stop(timeout=5)
            container.remove(force=True)

    def cleanup_tmpdir(self, tmpdir: str):
        """Remove temporary directory."""
        import shutil

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
