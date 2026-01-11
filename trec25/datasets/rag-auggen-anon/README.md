---
configs:
- config_name: inputs
  data_files:
  - split: test
    path: ["runs/*.jsonl"]
- config_name: truths
  data_files:
  - split: test
    path: ["trec-leaberboard.txt"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/trec-auto-judge/auto-judge-code/tree/main/trec25/judges/naive
    command: /naive-baseline.py --rag-responses $inputDataset --output $outputDir/trec-leaderboard.txt
    format:
      name: ["trec-eval-leaderboard"]
  input_format:
    name: "trec-rag-runs"
  truth_format:
    name: "trec-eval-leaderboard"
  evaluator:
    image: ghcr.io/trec-auto-judge/auto-judge-code/cli:0.0.1
    command: trec-auto-judge evaluate --input ${inputRun}/trec-leaderboard.txt --aggregate --output ${outputDir}/evaluation.prototext
ir_dataset:
  ir_datasets_id: "msmarco-segment-v2.1/trec-rag-2025"
---

# TREC RAG AUGGEN Submissions


# Internal: Construction of datasets:

On Gammaweb02 as root

```
cd /mnt/ceph/storage/data-tmp/current/kibi9872/auto-judge-code

sudo mount /mnt/ceph/tira
export IR_DATASETS_HOME=/mnt/ceph/tira/state/ir_datasets/
pip3 install ir-datasets --break-system-packages
pip3 uninstall -y ir-datasets --break-system-packages

PYTHONPATH=/mnt/ceph/storage/data-tmp/current/kibi9872/ir-datasets-rag-2025 trec-auto-judge export-corpus trec25/datasets/rag-auggen-anon/
```
