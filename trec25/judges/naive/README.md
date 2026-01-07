# Naive AutoJudge

This is the code for a naive auto-judge assessor that provides random or naive judgments. Our goal is that those judgments serve as naive baseline to indicate what minimal system ranking correlations could be for a target domain (e.g., similar to the "[Ranking Retrieval Systems without Relevance Judgments" paper](https://dl.acm.org/doi/abs/10.1145/383952.383961) that found that random relevance judgments can yield 0.4ish system ranking correlations).

## Usage

The `./naive-baseline.py --help` command provides an overview of the usage. For instance, to process the spot-check dataset, please run:

```
./naive-baseline.py judge --rag-responses ../../datasets/spot-check-dataset/runs --output naive-judgments
```