FROM ubuntu:noble


RUN apt-get update \
	&& apt-get install -y git python3 openjdk-11-jdk python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt --break-system-packages \
	&& python3 -m spacy download en_core_web_sm --break-system-packages

