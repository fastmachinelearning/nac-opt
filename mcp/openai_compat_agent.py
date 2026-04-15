#!/usr/bin/env python3
"""
Run nac-opt tools against any OpenAI-compatible chat-completions endpoint.
"""

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tools import OPENAI_TOOLS, call_openai_tool


DEFAULT_SYSTEM_PROMPT = (
    "You are an agent for the nac-opt repository. "
    "Use tools when needed, prefer reading repository files before making assumptions, "
    "and keep answers concise and actionable."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="User prompt to send to the model.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tool-roundtrips", type=int, default=8)
    return parser.parse_args()


def _require(value: str | None, name: str) -> str:
    if value:
        return value
    raise SystemExit(f"Missing {name}. Set it via an argument or environment variable.")


def _assistant_message_to_dict(message) -> dict:
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
        )

    payload = {
        "role": "assistant",
        "content": message.content or "",
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def main() -> int:
    args = parse_args()
    model = _require(args.model, "OPENAI_MODEL/--model")
    api_key = _require(args.api_key, "OPENAI_API_KEY/--api-key")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]

    for _ in range(args.max_tool_roundtrips):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(_assistant_message_to_dict(message))

        if not message.tool_calls:
            sys.stdout.write((message.content or "").strip() + "\n")
            return 0

        for tool_call in message.tool_calls:
            try:
                tool_output = call_openai_tool(
                    tool_call.function.name,
                    tool_call.function.arguments or "{}",
                )
            except Exception as exc:
                tool_output = f"Tool error: {type(exc).__name__}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_output,
                }
            )

    raise SystemExit(
        f"Stopped after {args.max_tool_roundtrips} tool round-trips without a final answer."
    )


if __name__ == "__main__":
    raise SystemExit(main())
