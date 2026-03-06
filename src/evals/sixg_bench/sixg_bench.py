"""6G-Bench: MCQ benchmark for AI-native 6G network reasoning.

3,722 expert-validated multiple-choice questions across 30 tasks aligned
with 6G standardisation (3GPP, IETF, ETSI, ITU-T, O-RAN Alliance).

Source: https://github.com/maferrag/6G-Bench
Paper:  https://hf.co/papers/2602.08675
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from evals._utils import resolve_dataset

DEFAULT_TASK_FILTER = "all"
DEFAULT_DATASET = "GSMA/ot-lite"
DEFAULT_DATASET_NAME = "sixg_bench"
DEFAULT_SPLIT = "test"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
        metadata={
            "task_id": record.get("task_id"),
            "task_name": record.get("task_name"),
            "difficulty": record.get("difficulty"),
            "category": record.get("category"),
        },
    )


@task
def sixg_bench(
    task_id: str = DEFAULT_TASK_FILTER,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    full: bool = False,
) -> Task:
    """6G-Bench: 30-task MCQ benchmark for 6G network reasoning."""
    ds_path, ds_split = resolve_dataset(full, dataset_path, DEFAULT_DATASET, split)
    dataset = hf_dataset(
        ds_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=ds_split,
    )
    if task_id != DEFAULT_TASK_FILTER:
        dataset = dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata.get("task_id") == task_id
        )
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
    )
