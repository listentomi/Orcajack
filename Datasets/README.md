# Datasets

Benchmark content is not redistributed in this repository. Fetch each benchmark from its own source and place it here in the layout below.

| Benchmark | Source |
|-----------|--------|
| Finance Agent Benchmark | Bigeard et al., 2025 |
| ScienceAgentBench | Chen et al., 2025 |
| TravelPlanner | Xie et al., 2024 |

## Expected layout

```text
Datasets/<benchmark>/
  task.json                        # list of task objects
  split_42_40/                     # produced by split_dataset.py
    task.train.json
    task.test.json
    batch.train.json
    batch.test.json
    split_metadata.json
```

`task.json` is a JSON list; each entry needs a task description under either
`task_description` or `task_inst`:

```json
[
  {"id": 1, "task_description": "How has US Steel addressed its planned merger ...", "question_type": "Market Analysis"}
]
```

## Reproducing our splits

`split_42_40_balanced/split_metadata.json` records the seed, the test ratio and the
exact `train_indices` / `test_indices` we used, together with the shuffle formula:

```python
rng = random.Random(seed); indices = list(range(n)); rng.shuffle(indices)
n_test = max(1, int(n * ratio))
test_indices  = sorted(indices[:n_test])
train_indices = sorted(indices[n_test:])
```

Point `split_dataset.py` at the same `task.json` with the same `--split-seed` and
`--test-ratio` to regenerate identical splits.
