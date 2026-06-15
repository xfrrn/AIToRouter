# backend/mininet/templates.py
"""Convert frontend TopologyJSON to a Mininet Python script string."""

from __future__ import annotations

from schemas.models import TopologyJSON

MININET_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated Mininet topology — automated iperf measurement mode."""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel
import json, sys, time, os, re

class CustomTopo(Topo):
    def build(self):
        {% for dev in devices %}
        {{ dev.hostname }} = self.addHost('{{ dev.hostname }}', ip='{{ dev.ip }}')
        {% endfor %}
        {% for conn in connections %}
        self.addLink({{ conn.src }}, {{ conn.dst }}, bw={{ conn.bw }}, delay='{{ conn.delay }}ms')
        {% endfor %}

def parse_iperf_bandwidth(output):
    """Extract measured bandwidth (Mbps) from iperf client output."""
    # iperf3: "receiver" line ends with "... Mbits/sec"
    # iperf2: last line with "Mbits/sec" or "MBytes"
    for line in reversed(output.splitlines()):
        m = re.search(r'([0-9.]+)\s*(M|G)bits/sec', line)
        if m:
            val = float(m.group(1))
            if m.group(2) == 'G':
                val *= 1000
            return val
        m = re.search(r'([0-9.]+)\s*(M|G)Bytes', line)
        if m:
            val = float(m.group(1)) * 8
            if m.group(2) == 'G':
                val *= 1000
            return val
    return None

def parse_ping_rtt(output):
    """Extract average RTT (ms) from ping output."""
    # e.g. "rtt min/avg/max/mdev = 0.123/0.456/0.789/0.123 ms"
    for line in output.splitlines():
        m = re.search(r'rtt min/avg/max/mdev =\s+[\d.]+/([\d.]+)/', line)
        if m:
            return float(m.group(1))
    return None

def run_iperf_flows(net, flows):
    """Run iperf flows and collect measured throughput."""
    results = []
    for flow in flows:
        src_name = flow["src"]
        dst_name = flow["dst"]
        bw = flow["bw_req"]
        duration = flow.get("duration", 5)
        flow_id = flow["flow_id"]

        src_host = net.get(src_name)
        dst_host = net.get(dst_name)
        if not src_host or not dst_host:
            results.append({"flow_id": flow_id, "src": src_name, "dst": dst_name,
                           "bw_req": bw, "measured_bw": 0, "error": "host not found"})
            continue

        # Start iperf server on dst
        dst_host.cmd("pkill iperf 2>/dev/null; iperf -s -p 5001 &")
        time.sleep(0.3)

        # Run iperf client
        out = src_host.cmd(
            f"iperf -c {dst_host.IP()} -p 5001 -b {bw}M -t {duration} 2>&1"
        )
        measured = parse_iperf_bandwidth(out)

        # Kill server
        dst_host.cmd("pkill iperf 2>/dev/null")

        results.append({
            "flow_id": flow_id, "src": src_name, "dst": dst_name,
            "bw_req": bw, "measured_bw": round(measured, 2) if measured else 0,
        })

    return results

def measure_link_rtt(net, edges):
    """Ping between every adjacent host pair to measure per-link RTT."""
    link_rtts = {}
    for edge in edges:
        s, d = edge["src"], edge["dst"]
        src_host = net.get(s)
        dst_host = net.get(d)
        if not src_host or not dst_host:
            continue
        out = src_host.cmd(f"ping -c 3 -q {dst_host.IP()} 2>&1")
        rtt = parse_ping_rtt(out)
        link_rtts[f"{s}-{d}"] = round(rtt, 2) if rtt else None
    return link_rtts

if __name__ == "__main__":
    setLogLevel("error")
    topo = CustomTopo()
    net = Mininet(topo=topo)
    net.start()
    time.sleep(2)

    # Quick connectivity check
    net.pingAll()

    flows_file = os.environ.get("FLOWS_FILE", "")
    flow_results = []
    link_rtts = {}

    if flows_file and os.path.exists(flows_file):
        with open(flows_file) as f:
            flows = json.load(f)
        flow_results = run_iperf_flows(net, flows)

    # Measure per-link RTT
    edges_json = os.environ.get("EDGES_JSON", "")
    if edges_json:
        edges = json.loads(edges_json)
        link_rtts = measure_link_rtt(net, edges)

    # Output structured result as a single JSON line with a unique marker
    output = {
        "flow_results": flow_results,
        "link_rtts": link_rtts,
    }
    print("MININET_RESULTS:" + json.dumps(output))
    sys.stdout.flush()

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
