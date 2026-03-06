"""NetConfEval: LLM benchmark for network configuration synthesis.

Four sub-tasks evaluating LLM capabilities on network configuration:
- formal_spec: Translate NL network requirements into JSON specifications
- conflict: Detect contradictions in network policy specifications
- routing_code: Generate Python routing algorithms
- config_gen: Generate FRRouting device configurations

Source: https://huggingface.co/datasets/NetConfEval/NetConfEval
Paper:  https://dl.acm.org/doi/10.1145/3656296
"""

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from evals._utils import resolve_dataset
from evals.netconfeval.utils import (
    conflict_scorer,
    frr_config_scorer,
    json_match_scorer,
)

TaskType = Literal["formal_spec", "conflict", "routing_code", "config_gen"]

DEFAULT_TASK_TYPE: TaskType = "formal_spec"
DEFAULT_DATASET = "GSMA/ot-lite"
DEFAULT_DATASET_NAME = "netconfeval"
DEFAULT_SPLIT = "test"


def record_to_sample(record: dict) -> Sample:
    """Convert a unified NetConfEval record to a Sample."""
    tt = record.get("task_type", "")
    if tt in ("formal_spec", "conflict"):
        return Sample(
            input=[
                ChatMessageSystem(content=record["description"]),
                ChatMessageUser(content=record["question"]),
            ],
            target=record["answer"],
            metadata={
                "task_type": tt,
                "n_policy_types": record.get("n_policy_types"),
                "batch_size": record.get("batch_size"),
                "conflict_exist": record.get("conflict_exist"),
            },
        )
    else:  # routing_code, config_gen
        return Sample(
            input=record["question"],
            target=record["answer"],
            metadata={
                "task_type": tt,
                "policy": record.get("policy"),
                "prompt_type": record.get("prompt_type"),
                "scenario_name": record.get("scenario_name"),
            },
        )


def _get_solver(task_type: TaskType):
    """Return the solver chain for a task type.

    formal_spec and conflict embed the original benchmark system prompt
    per-sample via ChatMessageSystem (from the dataset's ``description``
    field).  routing_code and config_gen have no separate system prompt —
    the dataset's ``prompt`` field already contains the full task
    instructions from the original multi-turn benchmark protocol.
    """
    return generate()


def _get_scorer(task_type: TaskType):
    """Return the scorer for a task type."""
    if task_type == "formal_spec":
        return json_match_scorer()
    elif task_type == "conflict":
        return conflict_scorer()
    elif task_type == "routing_code":
        # Routing code is scored by pattern match on function structure
        # (full test execution would require a sandbox)
        return includes()
    else:  # config_gen
        return frr_config_scorer()


@task
def netconfeval(
    task_type: TaskType = DEFAULT_TASK_TYPE,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    full: bool = False,
) -> Task:
    """NetConfEval: Network configuration synthesis benchmark."""
    ds_path, ds_split = resolve_dataset(full, dataset_path, DEFAULT_DATASET, split)

    dataset = hf_dataset(
        ds_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=ds_split,
    )

    dataset = dataset.filter(
        lambda sample: sample.metadata is not None
        and sample.metadata.get("task_type") == task_type
    )

    return Task(
        dataset=dataset,
        solver=_get_solver(task_type),
        scorer=_get_scorer(task_type),
    )
