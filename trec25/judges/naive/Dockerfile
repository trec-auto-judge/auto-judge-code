FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ENV OPENAI_BASE_URL=does-not-exist
ENV OPENAI_MODEL=does-not-exist
ENV OPENAI_API_KEY=does-not-exist

ADD naive-baseline.py /naive-baseline.py
