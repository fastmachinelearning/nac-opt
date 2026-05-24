"""
Heuristics for turning a plain-English request into planner constraints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

# The three boards rule4ml actually knows about (from supported_boards.json).
# Any other string passed as hls_config["board"] will raise ValueError in the estimator.
SUPPORTED_BOARDS: List[str] = ["pynq-z2", "zcu102", "alveo-u200", "alveo-u250"]

# Maps every reasonable human-language variant → canonical rule4ml board key.
# Sorted longest-first at module load so the first match wins (more specific beats less).
# More specific aliases (e.g. "alveo-u250") must appear before shorter ones ("alveo")
# so that the longest-first sort resolves ambiguity correctly.
BOARD_ALIASES: Dict[str, str] = {
    # pynq-z2
    "pynq-z2": "pynq-z2",
    "pynq z2": "pynq-z2",
    "pynq_z2": "pynq-z2",
    "pynqz2": "pynq-z2",
    "xc7z020": "pynq-z2",
    "pynq": "pynq-z2",
    "z2": "pynq-z2",
    # zcu102
    "zcu102": "zcu102",
    "zcu 102": "zcu102",
    "xczu9": "zcu102",
    # alveo-u250 (listed before alveo-u200 and generic "alveo" so longest match wins)
    "alveo-u250": "alveo-u250",
    "alveo u250": "alveo-u250",
    "alveo u 250": "alveo-u250",
    "alveo_u250": "alveo-u250",
    "alveou250": "alveo-u250",
    "xcu250": "alveo-u250",
    "u250": "alveo-u250",
    # alveo-u200
    "alveo-u200": "alveo-u200",
    "alveo u200": "alveo-u200",
    "alveo u 200": "alveo-u200",
    "alveo_u200": "alveo-u200",
    "alveou200": "alveo-u200",
    "xcu200": "alveo-u200",
    "u200": "alveo-u200",
    # generic "alveo" falls back to u200 (smaller/more common in tutorials)
    "alveo": "alveo-u200",
}

_SORTED_ALIASES = sorted(BOARD_ALIASES.keys(), key=len, reverse=True)


def _resolve_board(text: str) -> Optional[str]:
    """Return the canonical rule4ml board key for the first alias found in *text*, or None."""
    for alias in _SORTED_ALIASES:
        # Word-boundary guard: alias must not be immediately preceded/followed by [a-z0-9]
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return BOARD_ALIASES[alias]
    return None


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

    hardware_negations = (
        "no hardware",
        "without hardware",
        "no fpga",
        "without fpga",
        "skip hardware",
        "ignore hardware",
        "no hls",
        "no hardware constraints",
        "no hardware metrics",
    )
    hardware_keywords = (
        "latency",
        "resource",
        "hardware",
        "fpga",
        "board",
        "throughput",
        "cycles",
        "rule4ml",
        "hls4ml",
    )
    has_hw_negation = any(neg in text for neg in hardware_negations)
    if not has_hw_negation and any(kw in text for kw in hardware_keywords):
        constraints["use_hardware_metrics"] = True
        constraints["prefer_low_latency"] = True

    if "no local search" in text or "skip local search" in text:
        constraints["disable_local_search"] = True
    if "local search" in text and "no local search" not in text:
        constraints.setdefault("local_search", {})
        light_local_phrases = (
            "quick local",
            "fast local",
            "short local",
            "light local",
            "brief local",
            "small local",
            "quick qat",
            "smoke test the local",
            "smoke-test the local",
            "don't take too long",
            "keep it short",
            "keep local search short",
        )
        heavy_local_phrases = (
            "thorough local",
            "long local",
            "full local",
            "heavy local",
            "deep local",
            "exhaustive local",
        )
        if any(phrase in text for phrase in light_local_phrases):
            constraints["local_search"].setdefault("budget", "light")
        elif any(phrase in text for phrase in heavy_local_phrases):
            constraints["local_search"].setdefault("budget", "heavy")

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

    board = _resolve_board(text)
    if board:
        constraints.setdefault("hardware", {})
        constraints["hardware"]["board"] = board

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
