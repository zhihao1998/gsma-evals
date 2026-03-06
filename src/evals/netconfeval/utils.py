"""Custom scorers for NetConfEval sub-tasks."""

import json
import re

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from text, handling markdown code blocks."""
    # Try markdown code block first
    match = JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object/array in text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _normalize_value(v: object) -> object:
    """Normalize a value for comparison (lowercase strings, sort lists)."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return sorted(_normalize_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in v.items()}
    return v


@scorer(metrics=[accuracy(), stderr()])
def json_match_scorer():
    """Score formal specification translation by comparing JSON structures."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = _extract_json(state.output.completion)
        expected = _extract_json(target.text)

        if predicted is None:
            return Score(
                value=INCORRECT,
                answer="<no JSON found>",
                explanation="Could not parse JSON from model output",
            )

        if expected is None:
            return Score(
                value=INCORRECT,
                answer=str(predicted)[:200],
                explanation="Could not parse target JSON",
            )

        # Normalize and compare
        pred_norm = _normalize_value(predicted)
        exp_norm = _normalize_value(expected)
        is_correct = pred_norm == exp_norm

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=json.dumps(predicted)[:200],
            explanation=f"Match: {is_correct}",
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def conflict_scorer():
    """Score conflict detection by comparing JSON conflict specifications."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = _extract_json(state.output.completion)
        expected = _extract_json(target.text)

        if predicted is None:
            return Score(
                value=INCORRECT,
                answer="<no JSON found>",
                explanation="Could not parse JSON from model output",
            )

        if expected is None:
            return Score(
                value=INCORRECT,
                answer=str(predicted)[:200],
                explanation="Could not parse target JSON",
            )

        pred_norm = _normalize_value(predicted)
        exp_norm = _normalize_value(expected)
        is_correct = pred_norm == exp_norm

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=json.dumps(predicted)[:200],
            explanation=f"Match: {is_correct}",
        )

    return score


def _parse_frr_sections(config_text: str) -> dict[str, list[str]]:
    """Parse FRRouting config into sections keyed by device name."""
    devices: dict[str, list[str]] = {}
    current_device = None
    current_lines: list[str] = []

    for line in config_text.split("\n"):
        stripped = line.strip()
        # Device headers like "# Device: router1" or "hostname router1"
        if stripped.startswith("hostname "):
            if current_device and current_lines:
                devices[current_device] = current_lines
            current_device = stripped.split("hostname ", 1)[1].strip()
            current_lines = [stripped]
        elif stripped and current_device:
            current_lines.append(stripped)

    if current_device and current_lines:
        devices[current_device] = current_lines

    return devices


@scorer(metrics=[accuracy(), stderr()])
def frr_config_scorer():
    """Score FRRouting configuration generation by comparing config blocks."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion

        # Try to parse target as JSON (device -> config mapping)
        target_data = _extract_json(target.text)
        if isinstance(target_data, dict):
            # Target is JSON: {device_name: config_string, ...}
            target_devices = {
                k: v.strip().split("\n") if isinstance(v, str) else v
                for k, v in target_data.items()
            }
        else:
            # Target is raw config text
            target_devices = _parse_frr_sections(target.text)

        pred_devices = _parse_frr_sections(completion)

        if not target_devices:
            return Score(
                value=INCORRECT,
                answer="<parse error>",
                explanation="Could not parse target config",
            )

        if not pred_devices:
            return Score(
                value=INCORRECT,
                answer="<no config found>",
                explanation="Could not parse config from model output",
            )

        # Score as fraction of correctly matched devices
        total = len(target_devices)
        correct = 0
        for device, target_lines in target_devices.items():
            pred_lines = pred_devices.get(device, [])
            # Normalize: lowercase, strip whitespace
            target_set = {line.strip().lower() for line in target_lines if line.strip()}
            pred_set = {line.strip().lower() for line in pred_lines if line.strip()}
            if target_set == pred_set:
                correct += 1

        score_val = correct / total if total > 0 else 0
        is_correct = score_val == 1.0

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=f"{correct}/{total} devices correct",
            explanation=f"Matched {correct} of {total} device configs",
        )

    return score
