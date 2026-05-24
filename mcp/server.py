import sys
from pathlib import Path

from fastmcp import FastMCP

MCP_DIR = Path(__file__).resolve().parent
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

from tools import (
    _stdout_safe,
    create_search_config,
    describe_dataset,
    echo,
    inspect_dataset,
    list_available_boards,
    list_available_datasets,
    read_repo_file,
    read_search_results,
    recommend_search_plan,
    run_agentic_search,
    run_local_search,
    run_search_pipeline,
    run_search_pipeline_from_spec,
)

mcp = FastMCP(name="nac-opt-mcp")

for _fn in (
    echo,
    read_repo_file,
    list_available_datasets,
    list_available_boards,
    describe_dataset,
    inspect_dataset,
    recommend_search_plan,
    create_search_config,
    read_search_results,
    run_local_search,
    run_search_pipeline_from_spec,
    run_agentic_search,
    run_search_pipeline,
):
    mcp.tool(_stdout_safe(_fn))

if __name__ == "__main__":
    mcp.run()
