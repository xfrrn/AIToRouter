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
        m = re.search(r'([0-9.]+)\\s*(M|G)bits/sec', line)
        if m:
            val = float(m.group(1))
            if m.group(2) == 'G':
                val *= 1000
            return val
        m = re.search(r'([0-9.]+)\\s*(M|G)Bytes', line)
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
        m = re.search(r'rtt min/avg/max/mdev =\\s+[\\d.]+/([\\d.]+)/', line)
        if m:
            return float(m.group(1))
    return None

def run_iperf_flows(net, flows, direct_edges):
    """Run iperf flows in small concurrent batches and collect measured throughput.

    iperf v2 (the version available in the Mininet image) does not reliably
    start servers in background mode.  We use a serial per-flow approach:
    start server -> sleep briefly -> run client -> kill server -> next flow.
    This is slower but produces reliable measurements.
    """
    if not flows:
        return []

    BASE_PORT = 5201
    results = [None] * len(flows)

    for i, flow in enumerate(flows):
        flow_id = flow["flow_id"]
        src_name = flow["src"]
        dst_name = flow["dst"]
        bw = flow["bw_req"]
        duration = min(flow.get("duration", 3), 3)

        if direct_edges and (src_name, dst_name) not in direct_edges and (dst_name, src_name) not in direct_edges:
            results[i] = {"flow_id": flow_id, "src": src_name, "dst": dst_name,
                          "bw_req": bw, "measured_bw": None}
            continue

        src_host = net.get(src_name)
        dst_host = net.get(dst_name)
        if not src_host or not dst_host:
            results[i] = {"flow_id": flow_id, "src": src_name, "dst": dst_name,
                          "bw_req": bw, "measured_bw": None}
            continue

        port = BASE_PORT + i
        # Start server on destination, wait until port is listening
        dst_host.cmd(f"pkill -f 'iperf.*-p {port}' 2>/dev/null; true")
        dst_host.cmd(f"iperf -s -p {port} >/dev/null 2>&1 &")
        # Wait for server to be ready (retry up to 1s)
        src_host.cmd(f"for i in 1 2 3 4 5; do nc -z -w1 {dst_host.IP()} {port} && break; sleep 0.2; done; true")

        # Run client on source (iperf v2: use -f M for Mbps output)
        output = src_host.cmd(f"timeout {duration + 3} iperf -c {dst_host.IP()} -p {port} -b {bw}M -t {duration} -f M 2>&1")
        # Kill server for this port
        dst_host.cmd(f"pkill -f 'iperf.*-p {port}' 2>/dev/null; true")

        measured = parse_iperf_bandwidth(output)
        results[i] = {
            "flow_id": flow_id, "src": src_name, "dst": dst_name,
            "bw_req": bw,
            "measured_bw": round(measured, 2) if measured else None,
        }

    return results

def measure_link_rtt(net, edges):
    """Ping between every adjacent host pair to measure per-link RTT (parallel)."""
    if not edges:
        return {}
    # Launch all pings concurrently
    ping_pids = {}
    for edge in edges:
        s, d = edge["src"], edge["dst"]
        src_host = net.get(s)
        dst_host = net.get(d)
        if not src_host or not dst_host:
            continue
        out_file = f"/tmp/ping_{s}_{d}.txt"
        src_host.cmd(f"ping -c 3 -q {dst_host.IP()} > {out_file} 2>&1 & echo $!")
        ping_pids[f"{s}-{d}"] = (src_host, out_file)
    time.sleep(3)
    # Collect results
    link_rtts = {}
    for key, (src_host, out_file) in ping_pids.items():
        output = src_host.cmd(f"cat {out_file} 2>/dev/null; rm -f {out_file}")
        rtt = parse_ping_rtt(output)
        link_rtts[key] = round(rtt, 2) if rtt else None
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

    edges_json = os.environ.get("EDGES_JSON", "")
    edges = []
    if edges_json:
        edges = json.loads(edges_json)
    direct_edges = {(edge["src"], edge["dst"]) for edge in edges}

    if flows_file and os.path.exists(flows_file):
        with open(flows_file) as f:
            flows = json.load(f)
        flow_results = run_iperf_flows(net, flows, direct_edges)

    # Measure per-link RTT
    if edges:
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
