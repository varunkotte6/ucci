# Benchmarks

Public-data replications of the UCCI evaluation protocol (paper Section 6.1).
The paper's own workload is private and is not included in this repository.

| Directory | What it runs |
|---|---|
| [`conll2003/`](conll2003/README.md) | The replication proposed in the paper's Section 7: CoNLL-2003 English NER as JSON extraction, with a Qwen2.5 1.5B / 7B Instruct cascade, measured latency costs, the Table 2 baselines and ablations, and bootstrap intervals. |

Every number reported under a `runs/` directory is produced by the scripts
next to it, from the logs stored with it.
