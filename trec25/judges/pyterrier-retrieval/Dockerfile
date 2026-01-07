# The base image is build from https://github.com/OpenWebSearch/wows-code/blob/main/ecir26/template-new-approach/Dockerfile.dev
FROM ghcr.io/openwebsearch/wows-code/ecir-2026-pyterrier:latest

RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install git+https://github.com/trec-auto-judge/auto-judge-code.git

ENV OPENAI_BASE_URL=does-not-exist
ENV OPENAI_MODEL=does-not-exist
ENV OPENAI_API_KEY=does-not-exist

ADD retrieval-judge.py /retrieval-judge.py
