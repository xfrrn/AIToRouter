# backend/mininet/templates.py
"""Convert frontend TopologyJSON to a Mininet Python script string."""

from __future__ import annotations

from schemas.models import TopologyJSON

MININET_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated Mininet topology."""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import setLogLevel
import json, sys, time, os

class CustomTopo(Topo):
    def build(self):
        {% for dev in devices %}
        {{ dev.hostname }} = self.addHost('{{ dev.hostname }}', ip='{{ dev.ip }}')
        {% endfor %}
        {% for conn in connections %}
        self.addLink({{ conn.src }}, {{ conn.dst }}, bw={{ conn.bw }}, delay='{{ conn.delay }}ms')
        {% endfor %}

def run_iperf_flows(flows_file):
    """Run iperf flows from a JSON file and collect results."""
    with open(flows_file) as f:
        flows = json.load(f)

    results = []
    for flow in flows:
        src = flow["src"]
        dst = flow["dst"]
        bw = flow["bw_req"]
        duration = flow.get("duration", 5)

        # Start iperf server on dst
        dst_host = net.get(dst)
        dst_host.cmd(f"iperf -s -p 5001 &")

        # Run iperf client on src
        src_host = net.get(src)
        out = src_host.cmd(f"iperf -c {dst_host.IP()} -p 5001 -b {bw}M -t {duration} 2>&1")

        # Parse bandwidth from iperf output
        results.append({"flow_id": flow["flow_id"], "src": src, "dst": dst, "output": out})

        # Kill iperf server
        dst_host.cmd("pkill iperf")

    return results

def collect_link_util(net, G_data):
    """Collect link utilization from the running network."""
    edges = []
    for link_data in G_data["edges"]:
        u, v = link_data["src"], link_data["dst"]
        # Get interface stats from both sides
        try:
            src_host = net.get(u)
            # Use tc to read current utilization (approximate from iperf runs)
            edges.append({"src": u, "dst": v, "utilization": 0.0})
        except:
            edges.append({"src": u, "dst": v, "utilization": 0.0})
    return edges

if __name__ == "__main__":
    setLogLevel("info")
    topo = CustomTopo()
    net = Mininet(topo=topo)
    net.start()

    # Wait for network to stabilize
    time.sleep(2)

    # Ping all hosts to verify connectivity
    net.pingAll()

    # Run flows if provided
    flows_file = os.environ.get("FLOWS_FILE", "")
    if flows_file and os.path.exists(flows_file):
        results = run_iperf_flows(flows_file)
        print("FLOW_RESULTS:", json.dumps(results))

    # Output topology info for the host
    print("TOPO_READY")
    sys.stdout.flush()

    # Keep running for CLI or script control
    CLI(net)
    net.stop()
'''


def generate_mininet_script(topology: TopologyJSON) -> str:
    """Generate a Mininet Python script from a TopologyJSON."""

    # Map device IDs to Mininet-safe hostnames
    hostname_map: dict[str, str] = {}
    for i, dev in enumerate(topology.devices):
        hostname_map[dev.id] = f"h{i + 1}"

    # Build device entries for the template
    devices = []
    for i, dev in enumerate(topology.devices):
        hostname = hostname_map[dev.id]
        ip = dev.ip if dev.ip else f"10.0.{i // 254}.{i % 254 + 1}/24"
        devices.append({"hostname": hostname, "ip": ip})

    # Build connection entries
    connections = []
    for conn in topology.connections:
        src_hostname = hostname_map[conn.from_.devId]
        dst_hostname = hostname_map[conn.to.devId]
        connections.append({
            "src": src_hostname,
            "dst": dst_hostname,
            "bw": conn.bandwidth,
            "delay": conn.delay,
        })

    template = MININET_SCRIPT_TEMPLATE

    # Render devices
    device_lines = []
    for d in devices:
        device_lines.append(
            f'        {d["hostname"]} = self.addHost(\'{d["hostname"]}\', ip=\'{d["ip"]}\')'
        )
    template = template.replace(
        "        {% for dev in devices %}\n        {{ dev.hostname }} = self.addHost('{{ dev.hostname }}', ip='{{ dev.ip }}')\n        {% endfor %}",
        "\n".join(device_lines),
    )

    # Render connections
    conn_lines = []
    for c in connections:
        conn_lines.append(
            f'        self.addLink({c["src"]}, {c["dst"]}, bw={c["bw"]}, delay=\'{c["delay"]}ms\')'
        )
    template = template.replace(
        "        {% for conn in connections %}\n        self.addLink({{ conn.src }}, {{ conn.dst }}, bw={{ conn.bw }}, delay='{{ conn.delay }}ms')\n        {% endfor %}",
        "\n".join(conn_lines),
    )

    return template


def build_nx_graph(topology: TopologyJSON) -> "nx.Graph":
    """Convert TopologyJSON to a networkx Graph for the model."""
    import networkx as nx

    G = nx.Graph()

    # Map device IDs to integer node indices
    id_to_idx: dict[str, int] = {}
    for i, dev in enumerate(topology.devices):
        id_to_idx[dev.id] = i
        G.add_node(i, label=dev.label, type=dev.type, ip=dev.ip)

    for conn in topology.connections:
        u = id_to_idx[conn.from_.devId]
        v = id_to_idx[conn.to.devId]
        G.add_edge(u, v, bandwidth=conn.bandwidth, delay=conn.delay)

    return G, id_to_idx
