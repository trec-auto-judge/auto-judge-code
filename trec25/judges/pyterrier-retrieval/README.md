# PyTerrier Retrieval Judge

ToDo Description...

## Original Work

Idea is roughly from this paper:

```bibtex
@inproceedings{heineking:2025,
  author       = {Sebastian Heineking and Jonas Probst and Daniel Steinbach and Martin Potthast and Harrisen Scells},
  editor       = {Claudia Hauff and Craig Macdonald and Dietmar Jannach and Gabriella Kazai and Franco Maria Nardini and Fabio Pinelli and Fabrizio Silvestri and Nicola Tonellotto},
  title        = {Ranking Generated Answers - On the Agreement of Retrieval Models with Humans on Consumer Health Questions},
  booktitle    = {Advances in Information Retrieval - 47th European Conference on Information Retrieval, {ECIR} 2025, Lucca, Italy, April 6-10, 2025, Proceedings, Part {III}},
  series       = {Lecture Notes in Computer Science},
  volume       = {15574},
  pages        = {128--137},
  publisher    = {Springer},
  year         = {2025},
  doi          = {10.1007/978-3-031-88714-7\_10},
}

```

## Example Usage

```
./retrieval-judge.py --output leaderboard.trec --rag-responses ../../datasets/spot-check-dataset/runs/
```

