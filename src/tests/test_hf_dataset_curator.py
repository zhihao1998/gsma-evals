import json

from evals.hf_dataset_curator import (
    LITE_REPO_ID,
    build_repo_plan,
    parse_dataset_card,
    sample_sixg_lite,
)

LITE_README = """---
license: mit
pretty_name: Open Telco Sample Data
size_categories:
  - 1K<n<10K
configs:
  - config_name: teleqna
    data_files:
      - split: test
        path: test_teleqna.json
  - config_name: bins
    data_files:
      - split: test
        path: test_bins.json
dataset_info:
  - config_name: teleqna
    features:
      - name: question
        dtype: string
    splits:
      - name: test
        num_examples: 1000
  - config_name: bins
    features:
      - name: question
        dtype: string
    splits:
      - name: test
        num_examples: 100
---

# Open Telco Sample Data

## Citation

placeholder
"""


def test_sample_sixg_lite_balances_task_ids() -> None:
    records = [{"question": f"Q{i}", "task_id": "T1"} for i in range(3)] + [
        {"question": f"R{i}", "task_id": "T2"} for i in range(3)
    ]

    sampled = sample_sixg_lite(records, sample_size=4)

    counts = {"T1": 0, "T2": 0}
    for record in sampled:
        counts[record["task_id"]] += 1
    assert counts == {"T1": 2, "T2": 2}


def test_build_repo_plan_prunes_removed_configs_and_adds_sixg_lite() -> None:
    sixg_records = [
        {
            "question": "Question",
            "choices": ["A", "B"],
            "answer": 0,
            "task_id": "T1",
            "task_name": "Reasoning",
            "difficulty": "hard",
            "category": "standards",
        }
    ]

    plan = build_repo_plan(
        repo_id=LITE_REPO_ID,
        readme_text=LITE_README,
        repo_paths=("README.md", "test_bins.json", "test_teleqna.json"),
        sixg_lite_records=sixg_records,
    )

    assert plan.delete_paths == ("test_bins.json",)
    assert "test_sixg_bench.json" in plan.add_files
    assert json.loads(plan.add_files["test_sixg_bench.json"]) == sixg_records

    metadata, body = parse_dataset_card(plan.updated_readme)
    assert [config["config_name"] for config in metadata["configs"]] == [
        "teleqna",
        "sixg_bench",
    ]
    assert [info["config_name"] for info in metadata["dataset_info"]] == [
        "teleqna",
        "sixg_bench",
    ]
    assert 'load_dataset("GSMA/ot-lite", "sixg_bench", split="test")' in body
