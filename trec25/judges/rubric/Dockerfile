FROM ubuntu:jammy

RUN apt-get update \
	&& apt-get install -y git python3 python3-pip

RUN pip3 install uv

# Install main package from git with dspy extra
RUN uv pip install --system "git+https://github.com/trec-auto-judge/auto-judge-code.git#egg=trec_auto_judge[dspy]"

ADD rubric_autojudge.py /rubric_autojudge.py