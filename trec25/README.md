# Code for the Pilot iteration of TREC AutoJudge at TREC 25

This repository contains the datasets and approaches for the pilot iteration of the AutoJudge shared task.

## Datasets

TBD...

The RAG responses submitted to TREC are only available internally for TREC participants. We aim to build the shared code in this repository that one can directly load them in an ir-datasets compatible way when one copies the RAG responses that are only available to TREC participants to the corresponding locations.

## Approaches

We would be happy if you provide additional approaches via pull requests (we try to incorporate more and more approaches upon time).

Every approach is developed inside its own repository. We have some low-dependency baselines that you can look at to get minimal and naive examples:

- [naive](naive): A naive auto-judge assessor that provides random or naive judgments
- axiomatic: ToDo Maik + Heinrich ir-axioms/RAG-Axioms applied without modification
- ...

For every approach, we try to keep their CLI usage consistent so that they can be easily exchanges.

