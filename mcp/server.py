import sys
from pathlib import Path

from fastmcp import FastMCP

MCP_DIR = Path(__file__).resolve().parent
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

from tools import echo, read_repo_file, run_search_pipeline

mcp = FastMCP(name="nac-opt-mcp")

mcp.tool(echo)
mcp.tool(read_repo_file)
mcp.tool(run_search_pipeline)

if __name__ == "__main__":
    mcp.run()
