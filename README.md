# TREC AutoJudge (Meta-) Evaluation & Leaderboard


<p align="center">
   <img width=120px src="https://trec-auto-judge.cs.unh.edu/media/trec-auto-judge-logo-small.png">
   <br/>
   <br/>
   <a href="https://github.com/trec-auto-judge/auto-judge-code/actions/workflows/tests.yml">
   <img alt="Tests" src="https://github.com/trec-auto-judge/auto-judge-code/actions/workflows/tests.yml/badge.svg"/>
   </a>
   <a href="tests">
   <img alt="Coverage" src="tests/coverage.svg"/>
   </a>
   <br>
   <a href="https://trec-auto-judge.cs.unh.edu/">Web</a> &nbsp;|&nbsp;
   <a href="https://trec-auto-judge.cs.unh.edu/TREC_Auto_Judge.pdf">Proposal</a>
</p>

This repository contains the code used for evaluation and approaches for the TREC AutoJudge shared tasks.

- [trec25](trec25) the AutoJudge Pilot at TREC 2025 (**attention: this is work in progress**)
- [trec26](trec26) the upcoming iteration at TREC 2026 (**attention: this is work in progress**)

## What is TREC AutoJudge?


TREC Auto-Judge offers the first rigorous, cross-task benchmark for
Large-Language-Model judges.

While Large-Language-Model judges have emerged as a pragmatic solution when
manual relevance assessment is costly or infeasible. However, recent
studies reveal wide variation in accuracy across tasks, prompts, and
model sizes. 

Currently, shared task organizers choose an LLM judge per track ad
hoc, risking inconsistent baselines and hidden biases.

Auto-Judge provides a test bed for comparing different LLM judge ideas 
across several tasks and correlate results against manually created relevance 
judgments. AutoJudge proved a testbed to study emerging evaluation approaches, 
as well as vulnerabilities of LLM judges, and the efficacy of safeguards for 
those vulnerabilities. 

This Auto-Judge evaluation script standardizes data handling and evaluation 
across multiple shared tasks/TREC tracks that rely on LLM judging and 
provided a centralized, comparative evaluation of LLM judges under realistic 
conditions.



## What is this code for?

This project provides a means to evaluate AutoJudge approaches and provide a system ranking / leaderboard.

It will be used by TREC AutoJudge coordinators to score submissions. We encourage prospective participants to run this locally for method development.

This code will handle obtaining data sets (akin to `ir_datasets`), input/output and format conversions, and evaluation measures. 


# Code Setup

## Purpose

Initial code is in the [trec_auto_judge](trec_auto_judge) directory (**attention: this is in the very early brain storming phase**).


## Installation 

You can install the early prototype via:

```
pip3 install git+https://github.com/trec-auto-judge/auto-judge-code.git
```

## Rationale

After installing the code base, you will have to customize 

## Command Line Usage

After installation above, you can run `trec-auto-judge --help` which provides an overview of commands.


# Developer Section

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

