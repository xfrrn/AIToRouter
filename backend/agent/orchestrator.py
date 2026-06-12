# backend/agent/orchestrator.py
"""LLM Agent orchestrator using Claude API with function calling."""

from __future__ import annotations

import json

from anthropic import Anthropic

from schemas.models import TopologyJSON, DeploymentResult, ChatResponse

SYSTEM_PROMPT = """\
You are a network topology assistant for the AI Router platform. You help users design network topologies, generate traffic scenarios, deploy to Mininet, and interpret routing optimization results.

You have access to tools. Use them to fulfill the user's request step by step.

When generating a topology, the available device types are: router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer.
Connections between devices must have bandwidth (Mbps, default 100) and delay (ms, default 5).

When generating traffic, produce a realistic scenario matching the user's description. Each flow needs: flow_id, src (node index), dst (node index), bw_req (Mbps), phi (0-1, lower = delay-sensitive, higher = bandwidth-sensitive), duration (seconds).

When explaining results, compare the model's chosen path vs the OSPF path. Explain WHY the model made its choice in terms of link utilization and QoS tradeoffs. Be concise but insightful.
"""

TOOLS = [
    {
        "name": "generate_topology",
        "description": "Generate a network topology from a natural language description. Returns a TopologyJSON that will be loaded into the editor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the desired network topology. Include device types, counts, and how they connect."
                }
            },
            "required": ["description"]
        }
    },
    {
        "name": "generate_traffic",
        "description": "Generate traffic flows for a topology based on a scenario description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "num_nodes": {
                    "type": "integer",
                    "description": "Number of nodes in the topology."
                },
                "scenario": {
                    "type": "string",
                    "description": "Description of the traffic scenario (e.g. 'video conferencing', 'database replication', 'web browsing peak')."
                }
            },
            "required": ["num_nodes", "scenario"]
        }
    },
    {
        "name": "deploy_and_analyze",
        "description": "Deploy the topology to Mininet, run traffic, and get model-optimized routes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topology_json": {
                    "type": "string",
                    "description": "JSON string of the topology."
                },
                "traffic_json": {
                    "type": "string",
                    "description": "JSON string of the traffic flows array."
                }
            },
            "required": ["topology_json", "traffic_json"]
        }
    },
    {
        "name": "explain_results",
        "description": "Explain the routing optimization results in natural language.",
        "input_schema": {
            "type": "object",
            "properties": {
                "results_json": {
                    "type": "string",
                    "description": "JSON string of the deployment results."
                }
            },
            "required": ["results_json"]
        }
    },
]


class AgentOrchestrator:
    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6-20250514"

    def chat(
        self,
        message: str,
        topology: TopologyJSON | None = None,
        on_topology: callable = None,
        on_traffic: callable = None,
        on_deploy: callable = None,
    ) -> ChatResponse:
        """Process a user message through the Agent.

        Args:
            message: User's natural language message
            topology: Current editor topology state (if any)
            on_topology: Callback receiving TopologyJSON when agent generates one
            on_traffic: Callback receiving traffic list when agent generates one
            on_deploy: Callback receiving (topology, traffic) to trigger deploy

        Returns:
            ChatResponse with agent's reply and optional action data
        """
        messages = []

        # Build context about current state
        context = "The user is interacting with the AI Router topology editor."
        if topology:
            context += f"\nCurrent topology: {len(topology.devices)} devices, {len(topology.connections)} connections."
        context += "\n\nUser message: " + message

        messages.append({"role": "user", "content": context})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Process tool calls
        result_topology = None
        result_data = None
        reply = ""

        for block in response.content:
            if block.type == "text":
                reply += block.text
            elif block.type == "tool_use":
                tool_result = self._execute_tool(
                    block,
                    topology,
                    on_topology,
                    on_traffic,
                    on_deploy,
                )
                if tool_result:
                    if isinstance(tool_result, TopologyJSON):
                        result_topology = tool_result
                    elif isinstance(tool_result, DeploymentResult):
                        result_data = tool_result

        return ChatResponse(
            reply=reply.strip(),
            action="load_topology" if result_topology else ("show_results" if result_data else None),
            topology=result_topology,
            results=result_data,
        )

    def _execute_tool(
        self,
        tool_use,
        current_topology: TopologyJSON | None,
        on_topology: callable | None,
        on_traffic: callable | None,
        on_deploy: callable | None,
    ):
        name = tool_use.name
        inp = tool_use.input

        if name == "generate_topology":
            return self._tool_generate_topology(inp.get("description", ""), on_topology)

        elif name == "generate_traffic":
            return self._tool_generate_traffic(
                inp.get("num_nodes", 5), inp.get("scenario", ""), on_traffic
            )

        elif name == "deploy_and_analyze":
            return self._tool_deploy_and_analyze(
                inp.get("topology_json", "{}"),
                inp.get("traffic_json", "[]"),
                on_deploy,
            )

        elif name == "explain_results":
            return self._tool_explain_results(inp.get("results_json", "{}"))

        return None

    def _tool_generate_topology(self, description: str, on_topology) -> TopologyJSON | None:
        """Use LLM to generate a topology JSON from description."""
        prompt = f"""\
Generate a network topology as a JSON object based on this description: "{description}"

Available device types (use the exact type strings): router, switch, firewall, server, laptop, database, lb, cloud, wifi, printer

Return ONLY valid JSON in this exact format:
{{
  "devices": [
    {{"id": "dev-1", "type": "router", "x": 200, "y": 100, "label": "Core Router", "ip": "10.0.0.1"}}
  ],
  "connections": [
    {{"id": "conn-1", "from": {{"devId": "dev-1", "port": "bottom"}}, "to": {{"devId": "dev-2", "port": "top"}}, "bandwidth": 100, "delay": 5}}
  ]
}}

Rules:
- Position devices in a readable layout (x from 100-800, y from 50-500)
- Use ports "top", "right", "bottom", "left" appropriately for the layout
- Set reasonable bandwidth (10-10000 Mbps) and delay (0-100 ms) for each link
- Assign reasonable IPs (10.0.x.y)
- Each device needs a unique id starting from "dev-1"
- Each connection needs a unique id starting from "conn-1"
- Include ALL devices from the description"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        # Extract JSON from response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(text[json_start:json_end])
            topo = TopologyJSON(**data)
            if on_topology:
                on_topology(topo)
            return topo
        return None

    def _tool_generate_traffic(
        self, num_nodes: int, scenario: str, on_traffic
    ) -> list[dict]:
        """Generate traffic flows matching a scenario."""
        prompt = f"""\
Generate traffic flows for a network with {num_nodes} nodes based on this scenario: "{scenario}"

Return ONLY a JSON array of flow objects:
[
  {{"flow_id": 0, "src": 0, "dst": 3, "bw_req": 25.0, "phi": 0.3, "duration": 5}}
]

Rules:
- src and dst are integer node indices (0 to {num_nodes - 1}), src != dst
- bw_req: bandwidth requirement in Mbps (0.5 to 40)
- phi: QoS sensitivity (0 to 1). Lower = delay-sensitive (video, voice). Higher = bandwidth-sensitive (file transfer, backup)
- duration: flow duration in seconds (3 to 15)
- Generate 5-15 realistic flows matching the scenario
- Include a mix of phi values appropriate for the scenario"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            flows = json.loads(text[json_start:json_end])
            if on_traffic:
                on_traffic(flows)
            return flows
        return []

    def _tool_deploy_and_analyze(
        self,
        topology_json_str: str,
        traffic_json_str: str,
        on_deploy: callable | None,
    ) -> DeploymentResult | None:
        """Trigger deployment. Returns None (deploy is async in practice)."""
        if on_deploy:
            topology = TopologyJSON(**json.loads(topology_json_str))
            traffic = json.loads(traffic_json_str)
            return on_deploy(topology, traffic)
        return None

    def _tool_explain_results(self, results_json_str: str) -> str:
        """Use LLM to explain deployment results."""
        prompt = f"""\
Explain the following network routing optimization results in natural language.

For each flow, compare the model's chosen path vs the OSPF path. Explain WHY the model made its choice in terms of link utilization, delay, and QoS tradeoffs.

Results:
{results_json_str}

Be concise but insightful. Focus on the most interesting routing decisions."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
