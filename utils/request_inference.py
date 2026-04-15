"""
Heuristics for turning a plain-English request into planner constraints.
"""

from __future__ import annotations

from typing import Any, Dict
import re


def infer_constraints_from_request(request_text: str) -> Dict[str, Any]:
    text = (request_text or "").lower()
    constraints: Dict[str, Any] = {
        "open_ended": True,
    }
    requested_families = []

    if any(word in text for word in ("fast", "quick", "small test", "smoke test")):
        constraints["search_style"] = "fast"
    elif any(word in text for word in ("aggressive", "thorough", "deep search")):
        constraints["search_style"] = "aggressive"
    elif any(word in text for word in ("explore", "creative", "diverse")):
        constraints["search_style"] = "exploratory"
    else:
        constraints["search_style"] = "balanced"

    if any(word in text for word in ("latency", "resource", "hardware", "fpga", "board", "throughput", "cycles")):
        constraints["use_hardware_metrics"] = True
        constraints["prefer_low_latency"] = True

    if "no local search" in text or "skip local search" in text:
        constraints["disable_local_search"] = True
    if "local search" in text and "no local search" not in text:
        constraints.setdefault("local_search", {})

    if "mlp" in text:
        constraints["model_family"] = "mlp"
        requested_families.append("mlp")
    elif any(word in text for word in ("conv", "attention", "block")):
        constraints["model_family"] = "block"
        if "conv" in text:
            requested_families.append("conv")
        if "attention" in text or "attn" in text:
            requested_families.append("attention")

    if any(word in text for word in ("transformer", "attention", "attn")):
        constraints["prefer_attention"] = True
        constraints["prefer_expressive_models"] = True
        requested_families.append("transformer")

    if any(word in text for word in ("rnn", "lstm", "gru", "recurrent")):
        requested_families.append("rnn")

    if "deepsets" in text or "deep sets" in text:
        requested_families.append("deepsets")

    if any(phrase in text for phrase in ("best model", "best architecture", "highest accuracy", "best accuracy")):
        constraints["prefer_expressive_models"] = True

    if any(phrase in text for phrase in ("too slow", "too expensive", "too large", "keep it cheap", "low-cost", "low cost")):
        constraints["prefer_low_latency"] = True

    if "no attention" in text or "avoid attention" in text:
        constraints.setdefault("search_space_overrides", {})
        constraints["search_space_overrides"]["block_types"] = ["Conv", "MLP", "None"]
        constraints["avoid_attention"] = True

    trial_match = re.search(r"(\d+)\s+trials?", text)
    if trial_match:
        constraints["max_trials"] = int(trial_match.group(1))

    epoch_match = re.search(r"(\d+)\s+epochs?", text)
    if epoch_match:
        constraints["epochs"] = int(epoch_match.group(1))

    width_match = re.search(r"max(?:imum)?\s+width\s+(\d+)", text)
    if width_match:
        constraints["max_width"] = int(width_match.group(1))

    block_match = re.search(r"max(?:imum)?\s+blocks?\s+(\d+)", text)
    if block_match:
        constraints["max_blocks"] = int(block_match.group(1))

    board_match = re.search(r"\b(zcu102|vu13p|u250|u280)\b", text)
    if board_match:
        constraints.setdefault("hardware", {})
        constraints["hardware"]["board"] = board_match.group(1)

    latency_match = re.search(
        r"(?:latency(?:\s+budget)?|clock\s*cycles?|cycles?)\s*(?:under|below|<=|less than|at most|max(?:imum)?|target)?\s*([0-9][0-9_,.]*)",
        text,
    )
    if latency_match:
        raw_value = latency_match.group(1).replace(",", "").replace("_", "")
        constraints["latency_budget"] = float(raw_value)
        constraints["use_hardware_metrics"] = True
        constraints["prefer_low_latency"] = True

    resource_match = re.search(
        r"(?:resource(?:s)?(?:\s+budget)?|avg[_ ]resource)\s*(?:under|below|<=|less than|at most|max(?:imum)?|target)?\s*([0-9][0-9_,.]*)",
        text,
    )
    if resource_match:
        raw_value = resource_match.group(1).replace(",", "").replace("_", "")
        constraints["resource_budget"] = float(raw_value)
        constraints["use_hardware_metrics"] = True

    if requested_families:
        constraints["requested_model_families"] = sorted(set(requested_families))

    return constraints


__all__ = [
    "infer_constraints_from_request",
]
