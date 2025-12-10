# Submission Guidelines for TREC AutoJudge

We encourage [code submissions](#submission-variant-2-code-submissions) so that AutoJudge systems can be re-executed by others.
Organizing code so that it runs on machines that are not under your control requires more effort (please consider to add the code of your approach via a pull request to this repository as this allows us to provide help and maintain everything in one place).
We also allow to [manually upload submissions](#submission-variant-1-upload-of-submissions).

<details>
<summary>Prerequisite: Create an Account at TIRA.io and register a team to TREC AutoJudge in TIRA</summary>

### Step 1: Create an Account at TIRA

Please go to [https://www.tira.io/](https://www.tira.io/) and click on "Sign Up" to create a new account or "Log In" if you already have an account. You can either create an new account or Log in via GitHub or Google.

<img width="1042" height="965" alt="Screenshot_20251210_074005" src="https://github.com/user-attachments/assets/6f05d18d-3b03-4314-94b4-b1136613b362" />

### Step 2: Register Your Team to TREC AutoJudge

After you have logged in to TIRA, please navigate to the TREC AutoJudge task at [https://www.tira.io/task-overview/trec-auto-judge](https://www.tira.io/task-overview/trec-auto-judge). There, please click on "Register".

<img width="1726" height="554" alt="Screenshot_20251210_074553" src="https://github.com/user-attachments/assets/cb30f158-c62f-4201-9805-42dc1c0d64bb" />

### Step 3 (Optional): Manage your team

If you want to add others to your team, please navigate to your groups (under the hamburger menu) at [https://www.tira.io/g?type=my](https://www.tira.io/g?type=my)

</details>

## Submission Variant 1: Upload of Submissions

In cases where [code submissions](#submission-variant-2-code-submissions) do not make sense (e.g., for manually curated leaderboards, when systems are very experimental, or when deadlines are close), we encourage to upload your submissions manually.

A manual submission has the following two files:

```
├── ir-metadata.yml
└── trec-leaderboard.txt
```

where `ir-metadata.yml` describes the approach in the [ir_metadata format](https://www.ir-metadata.org/) (**attention: we still need to discuss which fields we want to make mandatory, currently nothing is mandatory**) and `trec-leaderboard.txt` is in a [format congruent to trec_eval -q](https://github.com/trec-auto-judge/auto-judge-code/tree/main/trec25/datasets/spot-check-dataset#formats).

The directory [leaderboard-upload-skeleton](leaderboard-upload-skeleton) contains an example that you can use as starter.

<details>
<summary>Step 1: Authentication and Login</summary>

We assume you have created an account at TIRA.io and have registered a team to TREC AutoJudge following the prerequisite above.

The preferred way to upload a submission to TIRA is via the command line interface, as this already can check that everything is in the correct format on your side.

Please install the TIRA cli via:

```
pip3 install --upgrade tira
```

Next, you need an authentication token:

- Navigate to the TREC AutoJudge task in TIRA [https://www.tira.io/task-overview/trec-auto-judge](https://www.tira.io/task-overview/trec-auto-judge)
- Click on "submit" => "Run Uploads" => "I want to upload runs via the command line". The UI shows your authentication token:

<img width="1964" height="503" alt="Screenshot_20251210_095119" src="https://github.com/user-attachments/assets/12e55ed2-a670-473c-ac4d-748a169afefa" />

Assuming your authentication token is AUTH-TOKEN, please authenticate via:

```
tira-cli login --token AUTH-TOKEN
```

Lastly, to verify that everything is correct, please run `tira-cli verify-installation`. Outputs might look like:

<img width="821" height="180" alt="Screenshot_20251210_095410" src="https://github.com/user-attachments/assets/51160132-eb19-4da3-8892-8a53adb41c71" />

</details>


<details>
<summary>Step 2: Upload your Submission</summary>

An complete overview of all dataset IDs for which you can upload submissions is available at [https://www.tira.io/datasets?query=trec-auto-judge](https://www.tira.io/datasets?query=trec-auto-judge). **Attention, some datasets that have missing responses or duplicated IDs are not yet available, as we first wanted to discuss how to handle them, this is ongoing in [this issue](https://github.com/trec-auto-judge/auto-judge-code/issues/2).**


Assuming you have your results in the `leaderboard-upload-skeleton` directory for the dataset id `spot-check-dataset-20251202-training`, then please first ensure that everything is valid via:

```
tira-cli upload --dataset spot-check-dataset-20251202-training --directory leaderboard-upload-skeleton --dry-run
```

The output should look like:

<img width="1074" height="131" alt="Screenshot_20251210_123926" src="https://github.com/user-attachments/assets/18c7f1d7-12d2-4ecc-9d2e-73cf31ec3582" />

If everything looks good, you can re-run the command and remove the `--dry-run` argument to upload your submission.

</details>

<details>
<summary>Alternative: Upload your Submission via the UI</summary>
TBD ...
</details>

## Submission Variant 2: Code Submissions

### Our Goal

**We want to make a reasonable set of LLM Judges easily accessible so that they can be easily re-executed.**

We would like to collect approaches via pull requests into this repository via this structure:



