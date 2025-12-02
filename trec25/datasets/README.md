# Datasets for the Pilot Round at TREC 2025

ToDo: Describe datasets.


# Admin Section

We use [TIRA](https://archive.tira.io) as backend for code submissions. The following describes how to upload a dataset to tira so that software submissions can run on the dataset.

## Step 1: Ensure your TIRA Client works

Install the tira client via:

```
pip3 install tira
```

Next, check that your TIRA client is correctly installed and that you are authenticated:

```
tira-cli verify-installation
```

If everything is as expected, the output should look like:

```
✓ You are authenticated against www.tira.io.
✓ TIRA home is writable.
✓ Docker/Podman is installed.
✓ The tirex-tracker works and will track experimental metadata.

Result:
✓ Your TIRA installation is valid.
```

## Step 2: Upload the Dataset

Assuming you have materialized the dataset as in the [spot-check-dataset](spot-check-dataset) datset example and you are authenticated against the TIRA backend as admin of the task, you can upload the dataset via:

```
tira-cli dataset-submission --path spot-check-dataset --task trec-auto-judge --split train --dry-run
```

This will check that the system-inputs and the truths are valid, it will run a baseline on it, will check that the outputs of the basline are valid and will run the evaluation on the baseline to ensure that everything works. If so, it will upload it to TIRA. All of this is configured in the README.md of the dataset directory in the Hugging Face datasets format.

If everything worked, the output should look like:

```
TIRA Dataset Submission:
✓ Your tira installation is valid.
✓ The configuration of the dataset spot-check-dataset is valid.
✓ The system inputs are valid.
✓ The truth data is valid.
✓ Repository for the baseline is cloned from https://github.com/trec-auto-judge/auto-judge-code.
✓ The baseline trec25/judges/naive is embedded in a Docker image.
✓ The baseline produced valid outputs at /tmp/tira-gdwth76l.
✓ The evaluation of the baseline produced valid outputs: {"Judges": 2.0, "Mean-Value": 5.9493926750674655, "Stdev-Value": 2.186709640458226}.
```

