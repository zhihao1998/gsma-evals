"""BINS: Business Intent and Network Slicing classification.

Classifies natural language business intents into 5G network slice types:
eMBB (enhanced Mobile Broadband), URLLC (Ultra-Reliable Low-Latency),
or mMTC (massive Machine-Type Communications).

Source: https://springernature.figshare.com/articles/dataset/27311538
Paper:  https://doi.org/10.1038/s41597-025-04736-z
"""

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message

from evals._utils import resolve_dataset

DEFAULT_SOURCE = "all"
DEFAULT_DATASET = "GSMA/ot-lite"
DEFAULT_DATASET_NAME = "bins"
DEFAULT_SPLIT = "test"

SYSTEM_PROMPT = """\
You are a 5G network slicing expert. Given a business intent description, \
classify it into exactly one network slice type:

- eMBB (enhanced Mobile Broadband): High bandwidth services like video \
streaming, web browsing, VR/AR, file downloads
- URLLC (Ultra-Reliable Low-Latency Communications): Mission-critical \
services like remote surgery, autonomous driving, industrial automation
- mMTC (massive Machine-Type Communications): IoT, sensors, smart meters, \
asset tracking with many low-power devices

Respond with ONLY the slice type: eMBB, URLLC, or mMTC"""

SLICE_PATTERN = re.compile(r"\b(eMBB|URLLC|mMTC)\b")


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer"],
        metadata={"source": record.get("source")},
    )


@scorer(metrics=[accuracy(), stderr()])
def slice_type_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        match = SLICE_PATTERN.search(completion)
        predicted = match.group(1) if match else ""
        is_correct = predicted == target.text
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=predicted,
            explanation=f"Predicted '{predicted}', target '{target.text}'",
        )

    return score


@task
def bins(
    source: str = DEFAULT_SOURCE,
    dataset_path: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    full: bool = False,
) -> Task:
    """BINS: Business intent to network slice classification."""
    ds_path, ds_split = resolve_dataset(full, dataset_path, DEFAULT_DATASET, split)
    dataset = hf_dataset(
        ds_path,
        name=DEFAULT_DATASET_NAME,
        sample_fields=record_to_sample,
        split=ds_split,
    )
    if source != DEFAULT_SOURCE:
        dataset = dataset.filter(
            lambda sample: sample.metadata is not None
            and sample.metadata.get("source") == source
        )
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=slice_type_scorer(),
    )
