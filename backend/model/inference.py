# backend/model/inference.py
"""network-rl Policy inference wrapper.

Follows the inference interface from 模型项目/network-rl/api introduction.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
import networkx as nx

# Add network-rl to path so we can import xchirl.
# The package root is 模型项目/network-rl/xchirl/ (where xchirl/ package lives).
_NETWORK_RL_ROOT = Path(__file__).resolve().parents[2] / "模型项目" / "network-rl" / "xchirl"
if str(_NETWORK_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETWORK_RL_ROOT))

from xchirl.utils.make_component_ppo import make_encoder
from xchirl.modules.encoders import PathPooler
from xchirl.modules.scorers import KPathScorer

# Normalization constants from api introduction.md
DELAY_MU, DELAY_SIG = 10.5, 5.5
BW_MU, BW_SIG = 65.0, 20.2


class Policy:
    """XCHiRL routing policy model (P4 mode, metrics dim=2)."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        data = torch.load(ckpt_path, weights_only=False, map_location=device)
        hp = data.get("hparams", {})

        hidden_dim = hp.get("hidden_dim", 256)
        kind = hp.get("encoder_kind", "film_gnn")

        self.encoder = make_encoder(
            hidden_dim, hp.get("layer_num", 4), kind=kind, heads=hp.get("heads", 1)
        )
        self.pooler = PathPooler(hidden_dim=hidden_dim)
        self.scorer = KPathScorer(hidden_dim=hidden_dim)

        sd = data["actor_state_dict"]
        self.encoder.load_state_dict(sd, strict=False)
        self.pooler.load_state_dict(sd, strict=False)
        self.scorer.load_state_dict(sd, strict=False)

        self.encoder.eval()
        self.pooler.eval()
        self.scorer.eval()
        self.to(device)

    def to(self, device):
        self.encoder.to(device)
        self.pooler.to(device)
        self.scorer.to(device)
        self._device = device
        return self

    @torch.no_grad()
    def forward(self, x, index, features, metrics, paths, paths_mask):
        """Select best path. Returns (action: int, logits: Tensor[K])."""
        h = self.encoder(x, index, features, metrics)
        h = self.pooler(h, paths, paths_mask)
        logits = self.scorer(h, metrics)
        return int(logits.argmax(dim=-1).item()), logits


class InferenceEngine:
    """Runs network-rl inference on a topology + flow list."""

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        if ckpt_path is None:
            ckpt_path = str(
                _NETWORK_RL_ROOT.parent / "runs" / "FILM_PPO" / "best.pt"
            )
        self.device = device
        self.policy: Policy | None = None
        self.ckpt_path = ckpt_path
        self.K = 16
        self.L_max = 22  # will be adjusted per topology

    def _ensure_loaded(self):
        if self.policy is None:
            if not Path(self.ckpt_path).exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {self.ckpt_path}. "
                    "Train a model first or provide a valid checkpoint path."
                )
            self.policy = Policy(self.ckpt_path, device=self.device)

    def infer(
        self, G: nx.Graph, flows: list[dict]
    ) -> tuple[list[dict], dict]:
        """Run inference on a list of flows over a topology.

        Args:
            G: networkx undirected graph with edge attrs 'bandwidth' and 'delay'
            flows: list of {flow_id, src, dst, bw_req, phi}

        Returns:
            (results, edge_utils) where results is a list of flow result dicts
            and edge_utils maps (u,v) -> utilization
        """
        self._ensure_loaded()

        N = G.number_of_nodes()
        self.L_max = N  # path buffer size = num nodes

        # Relabel nodes to consecutive integers
        G = nx.convert_node_labels_to_integers(G)

        # Build directed edge index [2, E]
        edges = []
        edge_bandwidths = {}
        edge_delays = {}
        for u, v in G.edges():
            edges += [(u, v), (v, u)]
            bw = G[u][v].get("bandwidth", 100.0)
            delay = G[u][v].get("delay", 5.0)
            edge_bandwidths[(u, v)] = bw
            edge_bandwidths[(v, u)] = bw
            edge_delays[(u, v)] = delay
            edge_delays[(v, u)] = delay

        index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)

        # Build edge features [E, 3]: delay_norm, util, bw_norm
        delay_norm_list, util_list, bw_norm_list = [], [], []
        for u, v in edges:
            d = edge_delays[(u, v)]
            c = edge_bandwidths[(u, v)]
            delay_norm_list.append((d - DELAY_MU) / DELAY_SIG)
            util_list.append(0.0)  # initial utilization
            bw_norm_list.append((c - BW_MU) / BW_SIG)

        features = torch.tensor(
            np.stack([delay_norm_list, util_list, bw_norm_list], axis=-1),
            dtype=torch.float32,
            device=self.device,
        )

        # Precompute K-shortest paths for each (src, dst) pair
        import itertools

        all_paths = {}
        for flow in flows:
            src, dst = flow["src"], flow["dst"]
            key = (src, dst)
            if key not in all_paths:
                paths_list = list(
                    itertools.islice(
                        nx.shortest_simple_paths(G, src, dst), self.K
                    )
                )
                all_paths[key] = paths_list

        results = []
        for flow in flows:
            src, dst = flow["src"], flow["dst"]
            bw_req = flow["bw_req"]
            phi = flow["phi"]

            paths_list = all_paths.get((src, dst), [])
            if not paths_list:
                results.append({
                    **flow,
                    "selected_path": [],
                    "hops": 0,
                    "ospf_path": [],
                    "max_link_utilization": 0.0,
                })
                continue

            # Build paths tensor [K, L]
            K_actual = min(len(paths_list), self.K)
            paths_tensor = torch.full(
                (self.K, self.L_max), -1, dtype=torch.long, device=self.device
            )
            paths_mask = torch.zeros(
                self.K, self.L_max, dtype=torch.bool, device=self.device
            )
            for k in range(K_actual):
                p = paths_list[k]
                paths_tensor[k, : len(p)] = torch.tensor(p, device=self.device)
                paths_mask[k, : len(p)] = True

            # Node features [N, 2]: [is_src, is_dst]
            x = torch.zeros(N, 2, device=self.device)
            x[src, 0] = 1.0
            x[dst, 1] = 1.0

            # Flow metrics [2]: phi, bw_req_norm
            metrics = torch.tensor(
                [phi, (bw_req - BW_MU) / BW_SIG],
                device=self.device,
            )

            # Inference
            action, logits = self.policy.forward(
                x, index, features, metrics, paths_tensor, paths_mask
            )

            if action < len(paths_list):
                selected = paths_list[action]
            else:
                selected = paths_list[0]

            # Compute OSPF (hop-shortest) path for comparison
            try:
                ospf_path = nx.shortest_path(G, src, dst, weight=None)
            except nx.NetworkXNoPath:
                ospf_path = []

            # Update edge utilization for this flow
            for i in range(len(selected) - 1):
                u, v = selected[i], selected[i + 1]
                for eid_dir in [(u, v), (v, u)]:
                    try:
                        eid = edges.index(eid_dir)
                        c = edge_bandwidths.get(eid_dir, 100.0)
                        current_util = features[eid, 1].item()
                        new_util = current_util + bw_req / c
                        features[eid, 1] = min(new_util, 1.0)
                    except ValueError:
                        pass

            # Compute max link utilization along selected path
            max_util = 0.0
            for i in range(len(selected) - 1):
                u, v = selected[i], selected[i + 1]
                for eid_dir in [(u, v), (v, u)]:
                    try:
                        eid = edges.index(eid_dir)
                        max_util = max(max_util, features[eid, 1].item())
                    except ValueError:
                        pass

            results.append({
                "flow_id": flow["flow_id"],
                "src": src,
                "dst": dst,
                "bw_req": bw_req,
                "phi": phi,
                "selected_path": selected,
                "hops": len(selected) - 1,
                "ospf_path": ospf_path,
                "max_link_utilization": round(max_util, 4),
            })

        # Return final edge utilization for display
        edge_utils = {}
        for i, (u, v) in enumerate(edges):
            if u < v:  # undirected, take one direction
                edge_utils[(u, v)] = features[i, 1].item()

        return results, edge_utils
