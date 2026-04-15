import sys
from pathlib import Path

from fastmcp import FastMCP

MCP_DIR = Path(__file__).resolve().parent
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

from tools import (
    create_search_config,
    describe_dataset,
    echo,
    inspect_dataset,
    list_available_datasets,
    read_repo_file,
    recommend_search_plan,
    run_agentic_search,
    run_search_pipeline,
    run_search_pipeline_from_spec,
)

mcp = FastMCP(name="nac-opt-mcp")

mcp.tool(echo)
mcp.tool(read_repo_file)
mcp.tool(list_available_datasets)
mcp.tool(describe_dataset)
mcp.tool(inspect_dataset)
mcp.tool(recommend_search_plan)
mcp.tool(create_search_config)
mcp.tool(run_search_pipeline_from_spec)
mcp.tool(run_agentic_search)
mcp.tool(run_search_pipeline)

if __name__ == "__main__":
    mcp.run()
