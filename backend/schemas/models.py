# backend/schemas/models.py
from __future__ import annotations

from pydantic import BaseModel, Field


class Device(BaseModel):
    id: str
    type: str  # router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer
    x: float
    y: float
    label: str
    ip: str = ""


class ConnectionEndpoint(BaseModel):
    devId: str
    port: str  # top, right, bottom, left


class Connection(BaseModel):
    id: str
    from_: ConnectionEndpoint = Field(alias="from")
    to: ConnectionEndpoint
    bandwidth: float = 100.0  # Mbps
    delay: float = 5.0  # ms

    class Config:
        populate_by_name = True


class TopologyJSON(BaseModel):
    devices: list[Device]
    connections: list[Connection]


class FlowResult(BaseModel):
    flow_id: int
    src: int
    dst: int
    bw_req: float
    phi: float
    selected_path: list[int]
    hops: int
    max_link_utilization: float
    ospf_path: list[int] | None = None
    ospf_max_link_utilization: float | None = None
    measured_bw: float | None = None  # Mbps, from Mininet iperf (None if Mininet not used)


class MininetFlowMeasurement(BaseModel):
    """Per-flow measurement from Mininet iperf."""
    flow_id: int
    src: str  # Mininet hostname, e.g. "h1"
    dst: str
    bw_req: float
    measured_bw: float | None  # actual iperf throughput in Mbps, None if not measurable


class DeploymentResult(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed"
    flows: list[FlowResult] = []
    error: str | None = None
    topology_nodes: list[int] = []
    topology_edges: list[dict] = []  # [{src, dst, bandwidth, delay, utilization}]
    mininet_used: bool = False
    mininet_flow_results: list[MininetFlowMeasurement] | None = None
    mininet_link_rtts: dict[str, float | None] | None = None  # {"src-dst": rtt_ms}


class ChatRequest(BaseModel):
    message: str
    topology: TopologyJSON | None = None  # current editor state, if any
    api_key: str | None = None  # Anthropic API key (user-provided)
    base_url: str | None = None  # Anthropic API base URL (for custom endpoints)


class ChatResponse(BaseModel):
    reply: str
    action: str | None = None  # "load_topology" | "show_results" | None
    topology: TopologyJSON | None = None
    results: DeploymentResult | None = None
