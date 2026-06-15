# backend/mininet/manager.py
"""Manage Mininet Docker container lifecycle."""

from __future__ import annotations

import uuid
import json
import os
import re
import tempfile
import logging
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound, DockerException

from schemas.models import TopologyJSON
from mininet.templates import generate_mininet_script, build_nx_graph

log = logging.getLogger("ai-router.mininet")
MININET_IMAGE = "iwaseyusuke/mininet:latest"
MININET_EXEC_TIMEOUT = 120  # seconds — iperf tests can take a while


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
        log.info("MininetManager initialized (Docker SDK connected)")

    def _ensure_image(self):
        """Pull Mininet image if not present. Called lazily on first deploy."""
        try:
            self.client.images.get(MININET_IMAGE)
            log.info("Mininet image already present: %s", MININET_IMAGE)
        except ImageNotFound:
            log.info("Pulling Mininet image %s ...", MININET_IMAGE)
            self.client.images.pull(MININET_IMAGE)
            log.info("Mininet image pulled successfully")

    def deploy(self, topology: TopologyJSON, flows: list[dict]) -> tuple[str, str, str, dict | None]:
        """Deploy topology to a Mininet container, run iperf, collect measurements.

        Returns (container_id, exec_id, tmpdir, mininet_results).
        mininet_results is None if the script failed or produced no output.
        """
        self._ensure_image()

        container_name = f"mininet-{uuid.uuid4().hex[:8]}"
        script = generate_mininet_script(topology)

        # Build edge list for per-link RTT measurement
        edges_for_env = []
        for conn in topology.connections:
            edges_for_env.append({"src": conn.from_.devId, "dst": conn.to.devId})

        # Write script and flows to temp directory
        tmpdir = tempfile.mkdtemp(prefix="mininet-")
        script_path = os.path.join(tmpdir, "topo.py")
        flows_path = os.path.join(tmpdir, "flows.json")

        with open(script_path, "w") as f:
            f.write(script)
        with open(flows_path, "w") as f:
            json.dump(flows, f)
        log.info("Generated Mininet script (%d bytes) + %d flows at %s",
                 len(script), len(flows), tmpdir)

        log.info("Creating container: %s (privileged)...", container_name)
        container = self.client.containers.run(
            MININET_IMAGE,
            name=container_name,
            command="tail -f /dev/null",
            volumes={tmpdir: {"bind": "/tmp/topo", "mode": "rw"}},
            environment={
                "FLOWS_FILE": "/tmp/topo/flows.json",
                "EDGES_JSON": json.dumps(edges_for_env),
            },
            privileged=True,
            detach=True,
            remove=False,
        )
        log.info("Container started: %s", container.id[:12])

        # Actually run the topology script and wait for completion
        log.info("Running Mininet topology script (timeout=%ds)...", MININET_EXEC_TIMEOUT)
        exit_code, output = container.exec_run(
            "python /tmp/topo/topo.py",
            stdout=True,
            stderr=True,
            demux=False,
        )
        stdout = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)

        log.info("Mininet script finished: exit_code=%d, output_len=%d", exit_code, len(stdout))

        # Parse structured results from stdout
        mininet_results = self._parse_results(stdout)
        if mininet_results:
            log.info("Mininet measurements: %d flow results, %d link RTTs",
                     len(mininet_results.get("flow_results", [])),
                     len(mininet_results.get("link_rtts", {})))
        else:
            log.warning("No MININET_RESULTS found in script output (first 300 chars): %s",
                        stdout[:300])

        return container.id, "", tmpdir, mininet_results

    def _parse_results(self, stdout: str) -> dict | None:
        """Extract MININET_RESULTS JSON from script stdout."""
        m = re.search(r'MININET_RESULTS:(\{.*\})', stdout, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

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
        log.info("Stopping container %s...", container_id[:12])
        container = self.get_container(container_id)
        if container:
            container.stop(timeout=5)
            container.remove(force=True)
            log.info("Container %s removed", container_id[:12])

    def cleanup_tmpdir(self, tmpdir: str):
        """Remove temporary directory."""
        import shutil

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
