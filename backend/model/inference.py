# backend/model/inference.py
"""network-rl Policy inference wrapper.

Follows the inference interface from 模型项目/network-rl/api introduction.md.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import networkx as nx

log = logging.getLogger("ai-router.model")

# Add network-rl to path so we can import xchirl.
# The package root is 模型项目/network-rl/xchirl/ (where xchirl/ package lives).
_NETWORK_RL_ROOT = Path(__file__).resolve().parents[2] / "模型项目" / "network-rl" / "xchirl"
if str(_NETWORK_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NETWORK_RL_ROOT))

# Lazy imports — xchirl depends on torch/torchrl which may not be installed.
# They are only needed when actually running inference.
_make_encoder = None
_PathPooler = None
_KPathScorer = None
_TOPOLOGY_CONFIGS = None


def _ensure_xchirl():
    global _make_encoder, _PathPooler, _KPathScorer, _TOPOLOGY_CONFIGS
    if _make_encoder is None:
        from xchirl.utils.make_component_ppo import make_encoder
        from xchirl.modules.encoders import PathPooler
        from xchirl.modules.scorers import KPathScorer
        from xchirl.envs.topology_configs import TOPOLOGY_CONFIGS
        _make_encoder = make_encoder
        _PathPooler = PathPooler
        _KPathScorer = KPathScorer
        _TOPOLOGY_CONFIGS = TOPOLOGY_CONFIGS

# Fallback normalization constants from api introduction.md.
DELAY_MU, DELAY_SIG = 10.5, 5.5
BW_MU, BW_SIG = 65.0, 20.2


def _strip_prefixed_state_dict(state_dict, prefix: str) -> dict:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _infer_metrics_dim(state_dict, hidden_dim: int) -> int:
    for key, value in state_dict.items():
        if key.endswith("to_features.0.weight") and value.ndim == 2:
            return int(value.shape[1] - hidden_dim)
    for key, value in state_dict.items():
        if key.endswith("films.0.to_gamma.0.weight") and value.ndim == 2:
            return int(value.shape[1])
    return 2


class Policy:
    """XCHiRL routing policy model."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torch
        _ensure_xchirl()

        data = torch.load(ckpt_path, weights_only=False, map_location=device)
        hp = data.get("hparams", {})

        hidden_dim = hp.get("hidden_dim", 256)
        kind = hp.get("encoder_kind", "film_gnn")
        heads = hp.get("heads", 1)
        topo_cfg = (_TOPOLOGY_CONFIGS or {}).get(hp.get("topo"), {})

        self.metrics_dim = _infer_metrics_dim(data["actor_state_dict"], hidden_dim)
        self.delay_mu = topo_cfg.get("DELAY_MU", DELAY_MU)
        self.delay_sig = topo_cfg.get("DELAY_SIG", DELAY_SIG)
        self.bw_mu = topo_cfg.get("BANDWIDTH_MU", BW_MU)
        self.bw_sig = topo_cfg.get("BANDWIDTH_SIG", BW_SIG)

        self.encoder = _make_encoder(
            hidden_dim, hp.get("layer_num", 4), kind=kind, heads=heads
        )
        self.pooler = _PathPooler(hidden_dim=hidden_dim, heads=heads)
        self.scorer = _KPathScorer(hidden_dim=hidden_dim)

        sd = data["actor_state_dict"]
        encoder_sd = _strip_prefixed_state_dict(sd, "module.0.module.0.module.")
        pooler_sd = _strip_prefixed_state_dict(sd, "module.0.module.1.module.")
        scorer_sd = _strip_prefixed_state_dict(sd, "module.0.module.2.module.")
        self.encoder.load_state_dict(encoder_sd or sd, strict=False)
        self.pooler.load_state_dict(pooler_sd or sd, strict=False)
        self.scorer.load_state_dict(scorer_sd or sd, strict=False)

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

    def forward(self, x, index, features, metrics, paths, paths_mask):
        """Select best path. Returns (action: int, logits: Tensor[K])."""
        import torch
        with torch.no_grad():
            h = self.encoder(x, index, features, metrics)
            h = self.pooler(h, paths, paths_mask)
            logits = self.scorer(h, metrics)
            return int(logits.argmax(dim=-1).item()), logits


class InferenceEngine:
    """Runs network-rl inference on a topology + flow list."""

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        if ckpt_path is None:
            # Look for checkpoints in project root runs/ first
            project_root = Path(__file__).resolve().parents[2]  # backend/../../
            ckpt_path = str(
                project_root / "runs" / "FILM_PPO" / "best.pt"
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
            import torch
            log.info("Loading network-rl checkpoint: %s", self.ckpt_path)
            self.policy = Policy(self.ckpt_path, device=self.device)
            log.info("Model loaded successfully on %s", self.device)

    def infer(
        self, G: nx.Graph, flows: list[dict]
    ) -> tuple[list[dict], dict]:
        self._ensure_loaded()
        log.info("Running inference: %d nodes, %d edges, %d flows",
                 G.number_of_nodes(), G.number_of_edges(), len(flows))

        import torch
        import numpy as np

        N = G.number_of_nodes()
        self.L_max = N  # path buffer size = num nodes

        # Relabel nodes to consecutive integers
        G = nx.convert_node_labels_to_integers(G)

        # Build directed edge index [2, E]
        edges = []
        edge_to_idx = {}
        edge_bandwidths = {}
        edge_delays = {}
        for u, v in G.edges():
            edge_to_idx[(u, v)] = len(edges)
            edges.append((u, v))
            edge_to_idx[(v, u)] = len(edges)
            edges.append((v, u))
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
            delay_norm_list.append((d - self.policy.delay_mu) / self.policy.delay_sig)
            util_list.append(0.0)  # initial utilization
            bw_norm_list.append((c - self.policy.bw_mu) / self.policy.bw_sig)

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

            bw_req_norm = (bw_req - self.policy.bw_mu) / self.policy.bw_sig
            if self.policy.metrics_dim == 3:
                delay_req = flow.get("delay_req", flow.get("delay", 100.0))
                delay_req_norm = (delay_req - self.policy.delay_mu) / self.policy.delay_sig
                metrics_values = [delay_req_norm, phi, bw_req_norm]
            else:
                metrics_values = [phi, bw_req_norm]
            metrics = torch.tensor(metrics_values, device=self.device)

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
                    eid = edge_to_idx.get(eid_dir)
                    if eid is not None:
                        c = edge_bandwidths.get(eid_dir, 100.0)
                        current_util = features[eid, 1].item()
                        new_util = current_util + bw_req / c
                        features[eid, 1] = min(new_util, 1.0)

            # Compute max link utilization along selected path
            max_util = 0.0
            for i in range(len(selected) - 1):
                u, v = selected[i], selected[i + 1]
                for eid_dir in [(u, v), (v, u)]:
                    eid = edge_to_idx.get(eid_dir)
                    if eid is not None:
                        max_util = max(max_util, features[eid, 1].item())

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
