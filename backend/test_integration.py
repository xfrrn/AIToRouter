# backend/test_integration.py
"""Integration test for the AI Router pipeline (no Docker or torch required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from schemas.models import TopologyJSON, Device, Connection, ConnectionEndpoint
from mininet.templates import build_nx_graph, generate_mininet_script
from traffic.generator import generate_flows


def test_topology_to_nx():
    """Test that TopologyJSON converts correctly to networkx Graph."""
    topo = TopologyJSON(
        devices=[
            Device(id="dev-1", type="router", x=100, y=100, label="R1", ip="10.0.0.1"),
            Device(id="dev-2", type="router", x=300, y=100, label="R2", ip="10.0.0.2"),
            Device(id="dev-3", type="server", x=200, y=300, label="S1", ip="10.0.1.1"),
        ],
        connections=[
            Connection(
                id="conn-1",
                from_=ConnectionEndpoint(devId="dev-1", port="bottom"),
                to=ConnectionEndpoint(devId="dev-3", port="top"),
                bandwidth=100,
                delay=5,
            ),
            Connection(
                id="conn-2",
                from_=ConnectionEndpoint(devId="dev-2", port="bottom"),
                to=ConnectionEndpoint(devId="dev-3", port="top"),
                bandwidth=50,
                delay=10,
            ),
        ],
    )

    G, id_to_idx = build_nx_graph(topo)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G[0][2]["bandwidth"] == 100
    assert G[1][2]["delay"] == 10
    assert id_to_idx["dev-1"] == 0
    assert id_to_idx["dev-3"] == 2
    print("PASS: topology_to_nx")


def test_flow_generation():
    """Test traffic flow generation."""
    flows = generate_flows(5, num_flows=10, seed=42)
    assert len(flows) == 10
    for f in flows:
        assert 0 <= f["src"] < 5
        assert 0 <= f["dst"] < 5
        assert f["src"] != f["dst"]
        assert 0.5 <= f["bw_req"] <= 40.0
        assert 0.0 <= f["phi"] <= 1.0
    print("PASS: flow_generation")


def test_mininet_script():
    """Test Mininet script generation."""
    topo = TopologyJSON(
        devices=[
            Device(id="dev-1", type="router", x=100, y=100, label="R1", ip="10.0.0.1"),
            Device(id="dev-2", type="server", x=200, y=200, label="S1", ip="10.0.0.2"),
        ],
        connections=[
            Connection(
                id="conn-1",
                from_=ConnectionEndpoint(devId="dev-1", port="bottom"),
                to=ConnectionEndpoint(devId="dev-2", port="top"),
                bandwidth=100,
                delay=5,
            ),
        ],
    )

    script = generate_mininet_script(topo)
    assert "class CustomTopo(Topo)" in script
    assert "h1 = self.addHost('h1'" in script
    assert "h2 = self.addHost('h2'" in script
    assert "bw=100" in script
    assert "delay='5.0ms'" in script
    print("PASS: mininet_script")


def test_inference_import():
    """Test that inference module can be imported (lazy — no torch required)."""
    from model.inference import InferenceEngine
    engine = InferenceEngine(ckpt_path="/nonexistent/path.pt")
    assert engine.ckpt_path == "/nonexistent/path.pt"
    assert engine.K == 16
    print("PASS: inference_import (lazy load works)")


def test_topology_json_parsing():
    """Test that frontend JSON format parses correctly with Pydantic."""
    # This simulates the JSON the frontend would send
    raw = {
        "devices": [
            {"id": "dev-1", "type": "router", "x": 100, "y": 100, "label": "R1", "ip": "10.0.0.1"},
            {"id": "dev-2", "type": "switch", "x": 300, "y": 100, "label": "SW1"},
        ],
        "connections": [
            {
                "id": "conn-1",
                "from": {"devId": "dev-1", "port": "bottom"},
                "to": {"devId": "dev-2", "port": "top"},
                "bandwidth": 50,
                "delay": 10,
            }
        ],
    }
    topo = TopologyJSON(**raw)
    assert len(topo.devices) == 2
    assert topo.connections[0].from_.devId == "dev-1"
    assert topo.connections[0].bandwidth == 50
    assert topo.connections[0].delay == 10
    print("PASS: topology_json_parsing")


if __name__ == "__main__":
    test_topology_to_nx()
    test_flow_generation()
    test_mininet_script()
    test_inference_import()
    test_topology_json_parsing()
    print("\nAll integration tests passed!")
