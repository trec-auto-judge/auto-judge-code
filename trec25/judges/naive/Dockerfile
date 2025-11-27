FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

