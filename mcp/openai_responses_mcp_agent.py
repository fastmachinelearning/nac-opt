#!/usr/bin/env python3
"""
Run the nac-opt MCP server through the OpenAI Responses API remote MCP tool.
"""

import argparse
import os
import sys

from openai import OpenAI


DEFAULT_INSTRUCTIONS = (
    "You are an agent for the nac-opt repository. "
    "Use the attached MCP server when needed, prefer reading repository files before "
    "making assumptions, and keep answers concise and actionable."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="User prompt to send to the model.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--server-url", default=os.environ.get("OPENAI_MCP_SERVER_URL"))
    parser.add_argument("--server-label", default=os.environ.get("OPENAI_MCP_SERVER_LABEL", "nac-opt"))
    parser.add_argument(
        "--server-description",
        default=os.environ.get(
            "OPENAI_MCP_SERVER_DESCRIPTION",
            "SNAC-Pack repository tools for reading files and running search pipelines.",
        ),
    )
    parser.add_argument(
        "--require-approval",
        choices=["always", "never"],
        default=os.environ.get("OPENAI_MCP_REQUIRE_APPROVAL", "never"),
    )
    parser.add_argument(
        "--allowed-tool",
        action="append",
        dest="allowed_tools",
        default=None,
        help="Limit remote MCP access to a specific tool name. Repeat to allow multiple tools.",
    )
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    return parser.parse_args()


def _require(value: str | None, name: str) -> str:
    if value:
        return value
    raise SystemExit(f"Missing {name}. Set it via an argument or environment variable.")


def main() -> int:
    args = parse_args()
    model = _require(args.model, "OPENAI_MODEL/--model")
    api_key = _require(args.api_key, "OPENAI_API_KEY/--api-key")
    server_url = _require(args.server_url, "OPENAI_MCP_SERVER_URL/--server-url")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    mcp_tool = {
        "type": "mcp",
        "server_label": args.server_label,
        "server_description": args.server_description,
        "server_url": server_url,
        "require_approval": args.require_approval,
    }
    if args.allowed_tools:
        mcp_tool["allowed_tools"] = args.allowed_tools

    response = client.responses.create(
        model=model,
        instructions=args.instructions,
        input=args.prompt,
        tools=[mcp_tool],
        max_output_tokens=args.max_output_tokens,
    )

    sys.stdout.write((response.output_text or "").strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
