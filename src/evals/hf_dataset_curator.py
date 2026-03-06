"""Curate the published Open Telco datasets on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

FULL_REPO_ID = "GSMA/ot-full"
LITE_REPO_ID = "GSMA/ot-lite"
README_PATH = "README.md"
DEFAULT_SIXG_LITE_SIZE = 150
DEFAULT_SIXG_SOURCE = FULL_REPO_ID
DEFAULT_SIXG_SOURCE_SPLIT = "test"
SIZE_BUCKET_SMALL = 1_000
SIZE_BUCKET_MEDIUM = 10_000
SIZE_BUCKET_LARGE = 100_000
REMOVE_CONFIGS = frozenset({"bins", "netconfeval"})
BENCHMARK_ORDER = (
    "teleqna",
    "teletables",
    "telemath",
    "telelogs",
    "3gpp_tsg",
    "oranbench",
    "srsranbench",
    "sixg_bench",
)
BENCHMARK_DETAILS = {
    "teleqna": (
        "Multiple-choice Q&A on telecom standards",
        "https://arxiv.org/abs/2310.15051",
    ),
    "teletables": (
        "Table interpretation from 3GPP specs",
        "https://arxiv.org/abs/2601.04202",
    ),
    "telemath": ("Telecom mathematical reasoning", "https://arxiv.org/abs/2506.10674"),
    "telelogs": ("5G network root cause analysis", "https://arxiv.org/abs/2507.21974"),
    "3gpp_tsg": (
        "3GPP document classification by working group",
        "https://arxiv.org/abs/2407.09424",
    ),
    "oranbench": (
        "Multiple-choice Q&A on O-RAN specifications",
        "https://arxiv.org/abs/2407.06245",
    ),
    "srsranbench": (
        "Multiple-choice Q&A on srsRAN 5G codebase",
        "https://arxiv.org/abs/2407.06245",
    ),
    "sixg_bench": (
        "AI-native 6G network reasoning",
        "https://arxiv.org/abs/2602.08675",
    ),
}
SIXG_LITE_PATH = "test_sixg_bench.json"
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class RepoPlan:
    repo_id: str
    updated_readme: str
    delete_paths: tuple[str, ...]
    add_files: dict[str, str]


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_sixg_records(source: str, split: str) -> list[dict[str, Any]]:
    """Load sixg_bench records from a dataset repo or local JSONL file."""
    source_path = Path(source)
    if source_path.exists():
        return load_jsonl_records(source_path)
    dataset = load_dataset(source, "sixg_bench", split=split)
    return [dict(record) for record in dataset]


def sample_sixg_lite(
    records: list[dict[str, Any]], sample_size: int
) -> list[dict[str, Any]]:
    """Select a deterministic, task-balanced subset for GSMA/ot-lite."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError("sample_size cannot exceed the number of available records")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_id") or "unknown")].append(record)

    for group in grouped.values():
        group.sort(
            key=lambda record: hashlib.md5(record["question"].encode()).hexdigest(),
        )

    selected: list[dict[str, Any]] = []
    round_index = 0
    group_names = sorted(grouped)
    while len(selected) < sample_size:
        added = False
        for group_name in group_names:
            group = grouped[group_name]
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        round_index += 1

    if len(selected) != sample_size:
        raise ValueError("could not build the requested sixg_bench sample subset")
    return selected


def parse_dataset_card(readme_text: str) -> tuple[dict[str, Any], str]:
    """Split a dataset card into front matter and markdown body."""
    match = FRONT_MATTER_RE.match(readme_text)
    if match is None:
        raise ValueError("dataset README is missing YAML front matter")
    return yaml.safe_load(match.group(1)), match.group(2)


def extract_citation_section(body: str) -> str:
    """Preserve the existing citation section verbatim."""
    marker = "\n## Citation\n"
    if marker not in body:
        return ""
    return "## Citation\n" + body.split(marker, 1)[1].strip() + "\n"


def config_sample_count(dataset_info: dict[str, Any]) -> int:
    """Read the published sample count from dataset_info."""
    return int(dataset_info["splits"][0]["num_examples"])


def size_category(total_examples: int) -> str:
    """Map a total sample count to the dataset card size bucket."""
    if total_examples < SIZE_BUCKET_SMALL:
        return "n<1K"
    if total_examples < SIZE_BUCKET_MEDIUM:
        return "1K<n<10K"
    if total_examples < SIZE_BUCKET_LARGE:
        return "10K<n<100K"
    return "n>100K"


def sort_card_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort dataset card entries using the canonical benchmark order."""
    return sorted(
        entries, key=lambda entry: BENCHMARK_ORDER.index(entry["config_name"])
    )


def sixg_lite_config_entry() -> dict[str, Any]:
    """Create the dataset card config entry for the sixg_bench sample file."""
    return {
        "config_name": "sixg_bench",
        "data_files": [{"split": "test", "path": SIXG_LITE_PATH}],
    }


def sixg_lite_dataset_info(sample_count: int) -> dict[str, Any]:
    """Create the dataset_info entry for the sixg_bench sample subset."""
    return {
        "config_name": "sixg_bench",
        "features": [
            {"name": "question", "dtype": "string"},
            {"name": "choices", "list": "string"},
            {"name": "answer", "dtype": "int64"},
            {"name": "task_id", "dtype": "string"},
            {"name": "task_name", "dtype": "string"},
            {"name": "difficulty", "dtype": "string"},
            {"name": "category", "dtype": "string"},
        ],
        "splits": [{"name": "test", "num_examples": sample_count}],
    }


def updated_card_metadata(
    repo_id: str,
    metadata: dict[str, Any],
    sixg_lite_count: int,
) -> dict[str, Any]:
    """Drop removed configs and ensure sixg_bench is available where needed."""
    configs = [
        config
        for config in metadata["configs"]
        if config["config_name"] not in REMOVE_CONFIGS
    ]
    dataset_info = [
        info
        for info in metadata["dataset_info"]
        if info["config_name"] not in REMOVE_CONFIGS
    ]

    if repo_id == LITE_REPO_ID:
        if not any(config["config_name"] == "sixg_bench" for config in configs):
            configs.append(sixg_lite_config_entry())
        dataset_info = [
            info for info in dataset_info if info["config_name"] != "sixg_bench"
        ]
        dataset_info.append(sixg_lite_dataset_info(sixg_lite_count))

    total_examples = sum(config_sample_count(info) for info in dataset_info)
    return {
        **metadata,
        "pretty_name": (
            "Open Telco Full Benchmarks"
            if repo_id == FULL_REPO_ID
            else "Open Telco Sample Data"
        ),
        "size_categories": [size_category(total_examples)],
        "configs": sort_card_entries(configs),
        "dataset_info": sort_card_entries(dataset_info),
    }


def benchmark_rows(dataset_info: list[dict[str, Any]]) -> str:
    """Render the benchmark overview table."""
    rows = []
    for info in dataset_info:
        config_name = info["config_name"]
        task, paper_url = BENCHMARK_DETAILS[config_name]
        rows.append(
            f"| `{config_name}` | {config_sample_count(info):,} | {task} | [arXiv]({paper_url}) |"
        )
    return "\n".join(rows)


def render_body(
    repo_id: str,
    dataset_info: list[dict[str, Any]],
    citation_section: str,
) -> str:
    """Render a clean README body from the updated dataset metadata."""
    total_examples = sum(config_sample_count(info) for info in dataset_info)
    benchmark_count = len(dataset_info)
    active_configs = ", ".join(info["config_name"] for info in dataset_info)

    if repo_id == FULL_REPO_ID:
        title = "Open Telco Full Benchmarks"
        summary = (
            f"**{total_examples:,} telecom-specific evaluation samples** across "
            f"{benchmark_count} benchmarks — the complete evaluation suite for measuring telecom AI performance."
        )
        description = (
            f"Use this dataset for final, publishable results. For fast iteration during model development, "
            f"use [`{LITE_REPO_ID}`](https://huggingface.co/datasets/{LITE_REPO_ID})."
        )
        companion_label = "Sample Data"
        companion_repo = LITE_REPO_ID
        note = f"> For quick testing, use [`{LITE_REPO_ID}`](https://huggingface.co/datasets/{LITE_REPO_ID})."
        eval_example = "uv run inspect eval src/evals/sixg_bench/sixg_bench.py --model openai/gpt-4o -T full=true"
    else:
        title = "Open Telco Sample Data"
        summary = (
            f"**{total_examples:,} telecom-specific evaluation samples** across "
            f"{benchmark_count} benchmarks — designed for fast iteration during model development."
        )
        description = (
            f"Use this dataset for fast iteration during model development. Evaluate against "
            f"[`{FULL_REPO_ID}`](https://huggingface.co/datasets/{FULL_REPO_ID}) for final results."
        )
        companion_label = "Full Benchmarks"
        companion_repo = FULL_REPO_ID
        note = f"> For full-scale evaluation, use [`{FULL_REPO_ID}`](https://huggingface.co/datasets/{FULL_REPO_ID})."
        eval_example = "uv run inspect eval src/evals/sixg_bench/sixg_bench.py --model openai/gpt-4o"

    sections = [
        f"# {title}",
        summary,
        description,
        (
            f"[Eval Framework](https://github.com/gsma-labs/evals) | "
            f"[{companion_label}](https://huggingface.co/datasets/{companion_repo})"
        ),
        "## Benchmarks",
        "| Config | Samples | Task | Paper |",
        "|--------|--------:|------|-------|",
        benchmark_rows(dataset_info),
        note,
        "## Quick Start",
        "```python",
        "from datasets import load_dataset",
        "",
        f'ds = load_dataset("{repo_id}", "sixg_bench", split="test")',
        f"# Available configs: {active_configs}",
        "```",
        "",
        "Or run evaluations with [Inspect AI](https://inspect.aisi.org.uk/):",
        "",
        "```bash",
        eval_example,
        "```",
        "",
        "See [Running Evaluations](https://github.com/gsma-labs/evals/blob/main/docs/running-evaluations.md) for the full guide.",
    ]
    if citation_section:
        sections.extend(["", citation_section.strip()])
    return "\n\n".join(sections).strip() + "\n"


def format_dataset_card(metadata: dict[str, Any], body: str) -> str:
    """Render the final README with front matter."""
    front_matter = yaml.safe_dump(metadata, sort_keys=False).strip()
    return f"---\n{front_matter}\n---\n\n{body}"


def delete_path_for_removed_config(path: str) -> bool:
    """Detect repo files that belong to removed configs."""
    return any(
        path == f"test_{config}.json" or path.startswith(f"{config}/")
        for config in REMOVE_CONFIGS
    )


def build_repo_plan(
    repo_id: str,
    readme_text: str,
    repo_paths: tuple[str, ...],
    sixg_lite_records: list[dict[str, Any]],
) -> RepoPlan:
    """Create a dry-run plan for one dataset repository."""
    metadata, body = parse_dataset_card(readme_text)
    updated_metadata = updated_card_metadata(repo_id, metadata, len(sixg_lite_records))
    updated_readme = format_dataset_card(
        updated_metadata,
        render_body(
            repo_id, updated_metadata["dataset_info"], extract_citation_section(body)
        ),
    )
    add_files = {README_PATH: updated_readme}
    if repo_id == LITE_REPO_ID:
        add_files[SIXG_LITE_PATH] = (
            json.dumps(
                sixg_lite_records,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    return RepoPlan(
        repo_id=repo_id,
        updated_readme=updated_readme,
        delete_paths=tuple(
            sorted(path for path in repo_paths if delete_path_for_removed_config(path))
        ),
        add_files=add_files,
    )


def repo_file_paths(api: HfApi, repo_id: str) -> tuple[str, ...]:
    """List the current file paths for a dataset repository."""
    return tuple(
        sorted(
            entry.path
            for entry in api.list_repo_tree(
                repo_id=repo_id, repo_type="dataset", recursive=True
            )
            if hasattr(entry, "blob_id")
        )
    )


def repo_readme_text(repo_id: str, token: str | None) -> str:
    """Fetch the current dataset README from Hugging Face."""
    readme_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=README_PATH,
        token=token,
    )
    return Path(readme_path).read_text(encoding="utf-8")


def print_plan(plan: RepoPlan) -> None:
    """Print a short dry-run summary."""
    print(f"\n{plan.repo_id}")
    print(f"  delete: {len(plan.delete_paths)} file(s)")
    for path in plan.delete_paths:
        print(f"    - {path}")
    for path in sorted(plan.add_files):
        print(f"  upsert: {path}")


def apply_repo_plan(api: HfApi, plan: RepoPlan) -> None:
    """Push the planned dataset changes to Hugging Face."""
    operations = [
        CommitOperationDelete(path_in_repo=path) for path in plan.delete_paths
    ]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=path,
            path_or_fileobj=content.encode("utf-8"),
        )
        for path, content in sorted(plan.add_files.items())
    )
    api.create_commit(
        repo_id=plan.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Keep sixg_bench and remove bins/netconfeval",
    )


def main(argv: list[str] | None = None) -> int:
    """Plan or apply the Hugging Face dataset cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the changes to Hugging Face instead of printing the plan.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Required for --push if not already logged in.",
    )
    parser.add_argument(
        "--sixg-lite-source",
        default=DEFAULT_SIXG_SOURCE,
        help="Dataset repo id or local JSONL source used to build the ot-lite sixg_bench subset.",
    )
    parser.add_argument(
        "--sixg-lite-source-split",
        default=DEFAULT_SIXG_SOURCE_SPLIT,
        help="Split to use when loading sixg_bench from a dataset repo.",
    )
    parser.add_argument(
        "--sixg-lite-size",
        type=int,
        default=DEFAULT_SIXG_LITE_SIZE,
        help="Number of sixg_bench samples to publish to GSMA/ot-lite.",
    )
    args = parser.parse_args(argv)

    sixg_records = sample_sixg_lite(
        load_sixg_records(args.sixg_lite_source, args.sixg_lite_source_split),
        args.sixg_lite_size,
    )
    api = HfApi(token=args.token)
    plans = [
        build_repo_plan(
            repo_id=repo_id,
            readme_text=repo_readme_text(repo_id, args.token),
            repo_paths=repo_file_paths(api, repo_id),
            sixg_lite_records=sixg_records,
        )
        for repo_id in (FULL_REPO_ID, LITE_REPO_ID)
    ]

    for plan in plans:
        print_plan(plan)

    if not args.push:
        print("\nDry run only. Re-run with --push to publish the changes.")
        return 0

    for plan in plans:
        apply_repo_plan(api, plan)
    print("\nPublished dataset changes to Hugging Face.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
