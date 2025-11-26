# TREC AutoJudge Code


<p align="center">
   <img width=120px src="https://trec-car.cs.unh.edu/trec-auto-judge/media/trec-auto-judge-logo-small.png">
   <br/>
   <br/>
   <a href="https://github.com/trec-auto-judge/auto-judge-code/actions/workflows/tests.yml">
   <img alt="Tests" src="https://github.com/trec-auto-judge/auto-judge-code/actions/workflows/tests.yml/badge.svg"/>
   </a>
   <a href="tests">
   <img alt="Coverage" src="tests/coverage.svg"/>
   </a>
   <br>
   <a href="https://trec-car.cs.unh.edu/trec-auto-judge/">Web</a> &nbsp;|&nbsp;
   <a href="https://trec-car.cs.unh.edu/trec-auto-judge/TREC_Auto_Judge.pdf">Proposal</a>
</p>

This repository contains the code used for evaluation and approaches for the TREC AutoJudge shared tasks.

- [trec25](trec25) the AutoJudge Pilot at TREC 2025 (**attention: this is work in progress**)
- [trec26](trec26) the upcoming iteration at TREC 2026 (**attention: this is work in progress**)

## Shared Code

For code that is re-used among different situations (e.g., handling input/output stuff, evaluation, connection to ir_datasets, etc.) we develop code in the [trec_auto_judge](trec_auto_judge) directory (**attention: this is in the very early brain storming phase**).

You can install the early prototype via:

```
pip3 install git+https://github.com/trec-auto-judge/auto-judge-code.git
```

### Unit Tests

Run unit tests via:

```
PYTHONPATH=. pytest .
```

Create Badge (TODO: add this to CI):

```
PYTHONPATH=. python3 -m pytest --cov-report term --cov=trec_auto_judge
coverage report --data-file=.coverage > test-coverage
coverage-badge -o tests/coverage.svg
```

