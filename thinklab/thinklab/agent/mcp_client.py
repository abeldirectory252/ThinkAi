"""
MCP (Model Context Protocol) client.
Connects to external MCP servers and exposes their tools to the agent.
"""
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger("thinklab.agent.mcp")


class MCPClient:
    """Connect to an MCP server and discover + invoke tools."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def list_tools(self) -> List[dict]:
        """Discover available tools from the MCP server."""
        r = self.session.get(f"{self.endpoint}/tools")
        r.raise_for_status()
        return r.json().get("tools", [])

    def call_tool(self, tool_name: str, params: dict) -> dict:
        """Invoke a tool on the MCP server."""
        r = self.session.post(f"{self.endpoint}/tools/{tool_name}", json=params)
        r.raise_for_status()
        return r.json()

    def list_resources(self) -> List[dict]:
        """List available resources from MCP server."""
        r = self.session.get(f"{self.endpoint}/resources")
        r.raise_for_status()
        return r.json().get("resources", [])

    def read_resource(self, uri: str) -> dict:
        """Read a resource by URI."""
        r = self.session.get(f"{self.endpoint}/resources/read",
                             params={"uri": uri})
        r.raise_for_status()
        return r.json()


def connect_mcp_tools(agent, mcp_endpoints: List[str]):
    """Connect MCP servers and register their tools into the agent."""
    for ep in mcp_endpoints:
        try:
            client = MCPClient(ep)
            tools = client.list_tools()
            for tool in tools:
                name = f"mcp_{tool['name']}"
                agent.tools.register(
                    name,
                    lambda client=client, tn=tool["name"], **p: client.call_tool(tn, p),
                )
                logger.info("Registered MCP tool: %s from %s", name, ep)
        except Exception as e:
            logger.warning("Failed to connect MCP %s: %s", ep, e)
