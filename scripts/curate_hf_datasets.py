#!/usr/bin/env python3
"""Keep sixg_bench and remove bins/netconfeval from the published HF datasets."""

from evals.hf_dataset_curator import main

if __name__ == "__main__":
    raise SystemExit(main())
