# UMBRELA

UMBRELA implementation with DSPy. Generalized to include multi-part queries, with title query, user background and verbalized problem statement.

While the UMBRELA prompt is designed to work with GPT-4o, this implementation is LLM model agnostic.

The dependencies used here are currently in a private repository, to install them, please either reach out to Laura Dietz or Maik Fröbe to get an invitation to the currently private repository. After you have access to this repository, please clone it here:

```
git clone git@github.com:laura-dietz/scale25-crucible.git
```

The dev-container of this repository is then configured to do all installation steps after the repository is cloned.

## Original Work


UMBRELA original citation

```bibtex
@article{upadhyay2024umbrela,
  title={Umbrela: Umbrela is the (open-source reproduction of the) bing relevance assessor},
  author={Upadhyay, Shivani and Pradeep, Ronak and Thakur, Nandan and Craswell, Nick and Lin, Jimmy},
  journal={arXiv preprint arXiv:2406.06519},
  year={2024}
}
```

UMBRELA reproduction with other LLMs (like Llama3.3)

```bibtex
@inproceedings{farzi2025does,
  title={Does UMBRELA work on other LLMs?},
  author={Farzi, Naghmeh and Dietz, Laura},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={3214--3222},
  year={2025}
}
```


## Usage

`$participantRunDir` 
: directory with (anonymized) participant submissions to the RAG25 track

`$out`
: File to which the leaderboard will be written.

```bash
./trec25/judges/umbrela/umbrela-baseline.py \ 
--rag-responses $participantRunsDir \
--rag-topics trec25/datasets/rag25/trec_rag_2025_requests.jsonl \
--output $out
```
