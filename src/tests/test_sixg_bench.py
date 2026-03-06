import importlib

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample

from evals.sixg_bench.sixg_bench import record_to_sample, sixg_bench

sixg_module = importlib.import_module("evals.sixg_bench.sixg_bench")


def test_record_to_sample_builds_stable_id_and_metadata() -> None:
    record = {
        "question": "What is 6G?",
        "choices": ["A", "B", "C"],
        "answer": 1,
        "task_id": "T1",
        "task_name": "Reasoning",
        "difficulty": "hard",
        "category": "standards",
    }

    sample = record_to_sample(record)

    assert sample.id == "T1_0375d8"
    assert sample.target == "B"
    assert sample.metadata == {
        "task_id": "T1",
        "task_name": "Reasoning",
        "difficulty": "hard",
        "category": "standards",
    }


def test_sixg_bench_task_uses_standard_mcq_components(monkeypatch) -> None:
    dataset = MemoryDataset(
        [
            Sample(
                input="Question 1",
                choices=["A", "B"],
                target="A",
                metadata={"task_id": "T1"},
            ),
            Sample(
                input="Question 2",
                choices=["A", "B"],
                target="B",
                metadata={"task_id": "T2"},
            ),
        ]
    )
    monkeypatch.setattr(sixg_module, "hf_dataset", lambda *args, **kwargs: dataset)

    task = sixg_bench(task_id="T1")

    samples = list(task.dataset)
    assert len(samples) == 1
    assert samples[0].metadata == {"task_id": "T1"}
    assert (
        task.solver.__dict__["__registry_info__"].name == "inspect_ai/multiple_choice"
    )
    assert task.solver.__dict__["__solver_all_params__"]["cot"] is False
    assert task.scorer[0].__dict__["__registry_info__"].name == "inspect_ai/choice"


def test_sixg_bench_smoke_eval_with_mock_model(monkeypatch, tmp_path) -> None:
    dataset = MemoryDataset(
        [
            Sample(
                id="T1_example",
                input="Pick A",
                choices=["First", "Second"],
                target="A",
                metadata={"task_id": "T1"},
            )
        ]
    )
    monkeypatch.setattr(sixg_module, "hf_dataset", lambda *args, **kwargs: dataset)

    logs = inspect_eval(
        sixg_bench(),
        model="mockllm/model",
        limit=1,
        log_dir=str(tmp_path / "logs"),
    )

    assert logs[0].status == "success"
