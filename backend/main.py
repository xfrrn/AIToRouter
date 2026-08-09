# backend/main.py
"""AI Router — FastAPI backend."""

from __future__ import annotations

import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from schemas.models import (
    TopologyJSON,
    DeploymentResult,
    FlowResult,
    MininetFlowMeasurement,
    ChatRequest,
    ChatResponse,
)
from mininet.templates import build_nx_graph
from traffic.generator import generate_flows
from model.inference import InferenceEngine

# ── Logging ───────────────────────────────────────────────────────
# Explicit handler — logging.basicConfig is unreliable with uvicorn reload
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_fmt)

log = logging.getLogger("ai-router")
log.setLevel(logging.INFO)
log.handlers.clear()
log.addHandler(_log_handler)
log.propagate = False  # don't bubble to root (uvicorn interferes with root handlers)

# Quiet down noisy libs
logging.getLogger("docker").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Global state
mn_manager = None  # MininetManager — initialized only if Docker is available
inference_engine: InferenceEngine | None = None
docker_available = False
jobs: dict[str, DeploymentResult] = {}
MAX_MININET_NODES = 8


def _build_result(
    job_id: str,
    G,
    flow_results: list[dict],
    edge_utils: dict,
    mininet_used: bool = False,
    mininet_data: dict | None = None,
    id_to_idx: dict | None = None,
) -> DeploymentResult:
    """Build DeploymentResult from inference output, optionally enriched with Mininet data."""
    # Map Mininet measured_bw per flow_id
    measured_map: dict[int, float] = {}
    if mininet_data:
        for mfr in mininet_data.get("flow_results", []):
            measured = float(mfr.get("measured_bw") or 0)
            if measured > 0:
                measured_map[mfr["flow_id"]] = measured

    result_flows = []
    for fr in flow_results:
        measured_bw = measured_map.get(fr["flow_id"])
        result_flows.append(
            FlowResult(
                flow_id=fr["flow_id"],
                src=fr["src"],
                dst=fr["dst"],
                bw_req=fr["bw_req"],
                phi=fr["phi"],
                selected_path=fr["selected_path"],
                hops=fr["hops"],
                max_link_utilization=fr["max_link_utilization"],
                ospf_path=fr.get("ospf_path"),
                ospf_max_link_utilization=fr.get("ospf_max_link_utilization"),
                measured_bw=measured_bw,
            )
        )

    # Build MininetFlowMeasurement list
    mininet_flow_results = None
    if mininet_data:
        mininet_flow_results = []
        for mfr in mininet_data.get("flow_results", []):
            mininet_flow_results.append(
                MininetFlowMeasurement(
                    flow_id=mfr["flow_id"],
                    src=mfr.get("src", ""),
                    dst=mfr.get("dst", ""),
                    bw_req=mfr.get("bw_req", 0),
                    measured_bw=mfr.get("measured_bw"),
                )
            )

    # Convert link_rtts keys from "h1-h2" to "srcIdx-dstIdx" if we have the mapping
    mininet_link_rtts = None
    if mininet_data and id_to_idx:
        raw_rtts = mininet_data.get("link_rtts", {})
        if raw_rtts:
            # Build reverse map: hostname → idx
            host_to_idx = {f"h{idx + 1}": idx for dev_id, idx in id_to_idx.items()}
            mininet_link_rtts = {}
            for key, rtt in raw_rtts.items():
                parts = key.rsplit("-", 1)
                if len(parts) == 2:
                    src_idx = host_to_idx.get(parts[0])
                    dst_idx = host_to_idx.get(parts[1])
                    if src_idx is not None and dst_idx is not None:
                        mininet_link_rtts[f"{src_idx}-{dst_idx}"] = rtt

    topo_edges = []
    for u, v, data in G.edges(data=True):
        util = edge_utils.get((u, v), edge_utils.get((v, u), 0.0))
        edge_entry = {
            "src": u,
            "dst": v,
            "bandwidth": data.get("bandwidth", 100.0),
            "delay": data.get("delay", 5.0),
            "utilization": round(util, 4),
        }
        if "measured_rtt" in data:
            edge_entry["measured_rtt"] = data["measured_rtt"]
        topo_edges.append(edge_entry)

    return DeploymentResult(
        job_id=job_id,
        status="completed",
        flows=result_flows,
        topology_nodes=list(range(G.number_of_nodes())),
        topology_edges=topo_edges,
        mininet_used=mininet_used,
        mininet_flow_results=mininet_flow_results,
        mininet_link_rtts=mininet_link_rtts,
    )


def _run_ospf_baseline(G, flows: list[dict]) -> tuple[list[dict], dict]:
    """Run OSPF (hop-shortest path) baseline when no trained model is available."""
    import networkx as nx

    results = []
    edge_utils = {}
    # Initialize utilization
    for u, v in G.edges():
        edge_utils[(u, v)] = 0.0
        edge_utils[(v, u)] = 0.0

    for flow in flows:
        src, dst = flow["src"], flow["dst"]
        bw_req = flow["bw_req"]

        try:
            path = nx.shortest_path(G, src, dst, weight=None)
        except nx.NetworkXNoPath:
            path = []

        # Update utilization along path
        max_util = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cap = G[u][v].get("bandwidth", 100.0)
            edge_utils[(u, v)] = edge_utils.get((u, v), 0.0) + bw_req / cap
            edge_utils[(v, u)] = edge_utils.get((v, u), 0.0) + bw_req / cap
            max_util = max(max_util, edge_utils[(u, v)])

        results.append({
            "flow_id": flow["flow_id"],
            "src": src,
            "dst": dst,
            "bw_req": bw_req,
            "phi": flow["phi"],
            "selected_path": path,
            "hops": max(len(path) - 1, 0),
            "ospf_path": path,
            "ospf_max_link_utilization": round(min(max_util, 1.0), 4),
            "max_link_utilization": round(min(max_util, 1.0), 4),
        })

    return results, edge_utils


def _attach_ospf_metrics(G, flows: list[dict], flow_results: list[dict]) -> None:
    """Attach true OSPF path utilization to model results for fair charting."""
    ospf_results, _ = _run_ospf_baseline(G, flows)
    ospf_by_id = {item["flow_id"]: item for item in ospf_results}
    for result in flow_results:
        ospf = ospf_by_id.get(result["flow_id"])
        if ospf:
            result["ospf_path"] = ospf["selected_path"]
            result["ospf_max_link_utilization"] = ospf["max_link_utilization"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mn_manager, inference_engine, docker_available

    # Initialize inference engine (always available)
    inference_engine = InferenceEngine()

    # Check Docker availability (lightweight — no image pull)
    from mininet.manager import check_docker_available
    docker_available = check_docker_available()
    if docker_available:
        print("[OK] Docker is available — /api/deploy enabled.")
    else:
        print("[WARN] Docker not available — /api/deploy disabled. Use /api/infer.")

    yield

    # Cleanup on shutdown
    for job_id in list(jobs.keys()):
        if jobs[job_id].status == "running":
            jobs[job_id].status = "failed"
            jobs[job_id].error = "Server shutdown"


app = FastAPI(title="AI Router", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    topo_html = Path(__file__).resolve().parent.parent / "topology-editor.html"
    return FileResponse(topo_html)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "docker": docker_available,
        "model_loaded": inference_engine.policy is not None if inference_engine else False,
    }


@app.post("/api/infer")
async def infer(topology: TopologyJSON, seed: int | None = None) -> DeploymentResult:
    """Infer optimal routes from topology — no Docker required."""
    job_id = uuid.uuid4().hex[:12]

    log.info("=" * 50)
    log.info("[/api/infer] job=%s | devices=%d | connections=%d",
             job_id, len(topology.devices), len(topology.connections))

    try:
        G, id_to_idx = build_nx_graph(topology)
        N = G.number_of_nodes()
        link_bandwidths = [data.get("bandwidth", 100.0) for _, _, data in G.edges(data=True)]
        flows = generate_flows(N, seed=seed, link_bandwidths=link_bandwidths)
        log.info("[/api/infer] topology: %d nodes, %d edges | generated %d flows",
                 N, G.number_of_edges(), len(flows))

        # Try model inference, fall back to OSPF baseline
        try:
            log.info("[/api/infer] attempting network-rl inference...")
            flow_results, edge_utils = inference_engine.infer(G, flows)
            _attach_ospf_metrics(G, flows, flow_results)
            log.info("[/api/infer] >>> using network-rl MODEL <<<")
        except FileNotFoundError:
            log.info("[/api/infer] no checkpoint found, using OSPF baseline")
            flow_results, edge_utils = _run_ospf_baseline(G, flows)
            log.info("[/api/infer] >>> using OSPF BASELINE <<<")
        except ImportError:
            log.info("[/api/infer] torch not installed, using OSPF baseline")
            flow_results, edge_utils = _run_ospf_baseline(G, flows)
            log.info("[/api/infer] >>> using OSPF BASELINE <<<")

        result = _build_result(job_id, G, flow_results, edge_utils,
                              mininet_used=False, mininet_data=None, id_to_idx=id_to_idx)
        log.info("[/api/infer] done | %d flow results", len(result.flows))

    except Exception as e:
        log.exception("[/api/infer] FAILED: %s", e)
        result = DeploymentResult(job_id=job_id, status="failed", error=str(e))

    jobs[job_id] = result
    return result


@app.post("/api/deploy")
async def deploy(
    topology: TopologyJSON,
    seed: int | None = None,
    use_mininet: bool = False,
) -> DeploymentResult:
    """Deploy topology to Mininet, run traffic, infer optimal routes.
    Falls back to inference-only if Docker or Mininet is unavailable.
    """
    job_id = uuid.uuid4().hex[:12]
    log.info("=" * 50)
    log.info("[/api/deploy] job=%s | devices=%d | connections=%d",
             job_id, len(topology.devices), len(topology.connections))

    # Convert topology and generate flows (common path)
    G, id_to_idx = build_nx_graph(topology)
    N = G.number_of_nodes()
    link_bandwidths = [data.get("bandwidth", 100.0) for _, _, data in G.edges(data=True)]
    flows = generate_flows(N, seed=seed, link_bandwidths=link_bandwidths)
    log.info("[/api/deploy] topology: %d nodes, %d edges | %d flows",
             N, G.number_of_edges(), len(flows))

    # ── Try Mininet path ──
    used_mininet = False
    mininet_data = None
    container_id = None
    tmpdir = None

    if use_mininet and N > MAX_MININET_NODES:
        log.info("[/api/deploy] Mininet skipped: topology has %d nodes (limit=%d)", N, MAX_MININET_NODES)
        use_mininet = False

    if docker_available and use_mininet:
        global mn_manager
        from mininet.manager import MininetManager
        try:
            if mn_manager is None:
                log.info("[/api/deploy] initializing MininetManager (may pull image)...")
                mn_manager = MininetManager()

            mininet_flows = []
            for f in flows:
                mininet_flows.append({
                    "flow_id": f["flow_id"],
                    "src": f"h{f['src'] + 1}",
                    "dst": f"h{f['dst'] + 1}",
                    "bw_req": f["bw_req"],
                    "duration": f["duration"],
                })

            log.info("[/api/deploy] deploying to Mininet + running iperf...")

            # Run blocking Docker operations in a thread to avoid freezing the event loop
            def _run_mininet():
                return mn_manager.deploy(topology, mininet_flows)

            container_id, _, tmpdir, mininet_data = await asyncio.to_thread(_run_mininet)

            if mininet_data is not None:
                used_mininet = True
                log.info("[/api/deploy] Mininet measurements collected successfully")

                # Feed measured RTT back into graph edges for inference
                raw_rtts = mininet_data.get("link_rtts", {})
                if raw_rtts:
                    updated = 0
                    for key, rtt in raw_rtts.items():
                        if rtt is None:
                            continue
                        # key format: "h1-h2" → map to device indices
                        parts = key.rsplit("-", 1)
                        if len(parts) != 2:
                            continue
                        src_idx = None
                        dst_idx = None
                        for dev_id, idx in id_to_idx.items():
                            hostname = f"h{idx + 1}"
                            if hostname == parts[0]:
                                src_idx = idx
                            if hostname == parts[1]:
                                dst_idx = idx
                        if src_idx is not None and dst_idx is not None:
                            if G.has_edge(src_idx, dst_idx):
                                old_delay = G[src_idx][dst_idx].get("delay", 5.0)
                                # Blend: 70% measured + 30% configured (avoids outlier noise)
                                blended = round(rtt * 0.7 + old_delay * 0.3, 2)
                                G[src_idx][dst_idx]["delay"] = blended
                                G[src_idx][dst_idx]["measured_rtt"] = rtt
                                updated += 1
                    if updated:
                        log.info("[/api/deploy] updated %d edge delays with measured RTT", updated)
            else:
                log.warning("[/api/deploy] Mininet returned no measurements, using static topology")
        except Exception as e:
            log.warning("[/api/deploy] Mininet failed (%s), falling back to direct inference", e)
            used_mininet = False
    elif docker_available:
        log.info("[/api/deploy] Mininet skipped by request; running direct inference")

    # ── Run inference ──
    try:
        if used_mininet:
            log.info("[/api/deploy] running inference with Mininet-measured topology...")
        else:
            log.info("[/api/deploy] running direct inference (no Mininet)...")

        flow_results, edge_utils = inference_engine.infer(G, flows)
        _attach_ospf_metrics(G, flows, flow_results)
        log.info("[/api/deploy] >>> used: %s + network-rl MODEL <<<",
                 "Mininet DOCKER" if used_mininet else "DIRECT inference")
    except (FileNotFoundError, ImportError):
        log.info("[/api/deploy] model not available, using OSPF baseline")
        flow_results, edge_utils = _run_ospf_baseline(G, flows)
        log.info("[/api/deploy] >>> used: %s + OSPF BASELINE <<<",
                 "Mininet DOCKER" if used_mininet else "DIRECT inference")

    result = _build_result(
        job_id, G, flow_results, edge_utils,
        mininet_used=used_mininet,
        mininet_data=mininet_data,
        id_to_idx=id_to_idx,
    )
    log.info("[/api/deploy] done | %d flow results | mininet=%s",
             len(result.flows), used_mininet)

    # Cleanup
    if container_id is not None:
        log.info("[/api/deploy] cleaning up container=%s", container_id[:12])
        try:
            mn_manager.stop_and_remove(container_id)
        except Exception:
            pass
    if tmpdir is not None:
        try:
            mn_manager.cleanup_tmpdir(tmpdir)
        except Exception:
            pass

    jobs[job_id] = result
    return result


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> DeploymentResult:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.delete("/api/containers/{container_id}")
async def remove_container(container_id: str):
    if mn_manager is None:
        raise HTTPException(status_code=503, detail="Docker not available")
    mn_manager.stop_and_remove(container_id)
    return {"status": "removed", "container_id": container_id}


# ── Agent chat ────────────────────────────────────────────────────
agent = None  # AgentOrchestrator — initialized lazily when first used


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint — natural language driven orchestration."""
    global agent, inference_engine

    if agent is None:
        try:
            from agent.orchestrator import AgentOrchestrator
            agent = AgentOrchestrator()
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Agent is not available. Install 'openai' package and set API key in chat panel.",
            )

    def on_topology(topo):
        pass

    def on_traffic(flows):
        pass

    def on_deploy(topo, traffic):
        log.info("[/api/chat] agent triggered deploy_and_analyze")
        G, _ = build_nx_graph(topo)
        try:
            flow_results, edge_utils = inference_engine.infer(G, traffic)
            _attach_ospf_metrics(G, traffic, flow_results)
            log.info("[/api/chat] >>> using model inference <<<")
        except (FileNotFoundError, ImportError):
            flow_results, edge_utils = _run_ospf_baseline(G, traffic)
            log.info("[/api/chat] >>> using OSPF baseline <<<")
        return _build_result("agent", G, flow_results, edge_utils)

    try:
        log.info("[/api/chat] message=%s...", request.message[:60])
        result = agent.chat(
            request.message,
            topology=request.topology,
            on_topology=on_topology,
            on_traffic=on_traffic,
            on_deploy=on_deploy,
            api_key=request.api_key,
            base_url=request.base_url,
        )
        log.info("[/api/chat] reply=%s... action=%s", (result.reply or "")[:40], result.action)
        return result
    except Exception as e:
        log.exception("[/api/chat] FAILED: %s", e)
        return ChatResponse(reply=f"Agent 调用失败: {e}")
