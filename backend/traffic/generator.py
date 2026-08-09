# backend/traffic/generator.py
"""Auto-generate traffic flows for a given topology."""

from __future__ import annotations

import random


def generate_flows(
    num_nodes: int,
    num_flows: int | None = None,
    seed: int | None = None,
    link_bandwidths: list[float] | None = None,
) -> list[dict]:
    """Generate random flow demands for a topology.

    Matches the distribution used in network-rl training:
    - bw defaults to U[0.5, 40]
    - if link_bandwidths is provided, bw scales with the lower quartile capacity
    - phi ~ U[0, 1]
    - Each flow between a random (src, dst) pair, src != dst

    Returns list of {flow_id, src, dst, bw_req, phi, duration}.
    """
    rng = random.Random(seed)

    if num_flows is None:
        num_flows = min(num_nodes * 3, 50)  # reasonable default

    capacities = sorted(float(bw) for bw in (link_bandwidths or []) if float(bw) > 0)
    if capacities:
        ref_capacity = capacities[max(0, len(capacities) // 4)]
        min_bw = max(0.5, ref_capacity * 0.06)
        max_bw = max(min_bw + 0.5, ref_capacity * 0.30)
    else:
        min_bw, max_bw = 0.5, 40.0

    flows = []
    for flow_id in range(num_flows):
        src = rng.randint(0, num_nodes - 1)
        dst = rng.randint(0, num_nodes - 1)
        while dst == src:
            dst = rng.randint(0, num_nodes - 1)

        bw_req = round(rng.uniform(min_bw, max_bw), 2)
        phi = round(rng.uniform(0.0, 1.0), 2)
        duration = rng.randint(2, 4)

        flows.append({
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
            "bw_req": bw_req,
            "phi": phi,
            "duration": duration,
        })

    return flows
