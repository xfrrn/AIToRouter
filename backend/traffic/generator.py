# backend/traffic/generator.py
"""Auto-generate traffic flows for a given topology."""

from __future__ import annotations

import random


def generate_flows(
    num_nodes: int,
    num_flows: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Generate random flow demands for a topology.

    Matches the distribution used in network-rl training:
    - bw ~ U[0.5, 40] (matches FLOW_BW range)
    - phi ~ U[0, 1]
    - Each flow between a random (src, dst) pair, src != dst

    Returns list of {flow_id, src, dst, bw_req, phi, duration}.
    """
    if seed is not None:
        random.seed(seed)

    if num_flows is None:
        num_flows = min(num_nodes * 3, 50)  # reasonable default

    flows = []
    for flow_id in range(num_flows):
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        while dst == src:
            dst = random.randint(0, num_nodes - 1)

        bw_req = round(random.uniform(0.5, 40.0), 2)
        phi = round(random.uniform(0.0, 1.0), 2)
        duration = random.randint(3, 10)

        flows.append({
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
            "bw_req": bw_req,
            "phi": phi,
            "duration": duration,
        })

    return flows
