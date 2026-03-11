from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP(name="nac-opt-mcp") # makes mcp server

REPO_ROOT = Path(__file__).resolve().parents[1] # set to nac-opt directory

# repo root is on sys.path so can import utils
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@mcp.tool
def echo(text: str) -> str: # check it works with the prompt: Use the echo tool from nac-opt-mcp with text = "hello" and show me the result.
    """Echo text back to the caller."""
    return text

@mcp.tool
def read_repo_file(relative_path: str) -> str: # another check with the prompt: Use read_repo_file from nac-opt-mcp with relative_path = "README.md" and show me the first 40 lines.
    """Read a UTF-8 text file from this repo by relative path."""
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {relative_path}")
    return path.read_text(encoding="utf-8")


@mcp.tool # full pipeline with the prompt: Use run_search_pipeline from nac-opt-mcp with config_relative_path = "tutorials/tutorial_3_qubit/t3_config.yaml" and run_local_search = true, and show me the returned summary.
def run_search_pipeline(config_relative_path: str, run_local_search: bool = True) -> str:
    """
    Run the repo's existing global search + (optional) local search pipeline.

    Parameters
    ----------
    config_relative_path:
        Path to a tutorial-style YAML config, relative to the `nac-opt` directory.
        Example: "tutorials/tutorial_2_block/t2_config.yaml"
    run_local_search:
        If True, run local search after global search completes.

    Returns
    -------
    str
        A YAML string summarizing where artifacts were written (results dirs, YAML paths).
    """
    from utils.search_pipeline import run_pipeline_from_config

    config_path = (REPO_ROOT / config_relative_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"No such config file: {config_relative_path}")

    summary = run_pipeline_from_config(str(config_path), run_local_search=run_local_search)

    safe_summary = {
        "results_dir": summary.get("results_dir"),
        "architecture_yaml": summary.get("architecture_yaml"),
        "local_results_dir": summary.get("local_results_dir"),
    }
    return yaml_dump(safe_summary)


def yaml_dump(obj) -> str:
    import yaml

    return yaml.safe_dump(obj, sort_keys=False)

if __name__ == "__main__":
    mcp.run()