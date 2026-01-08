FROM ubuntu:jammy

RUN apt-get update \
	&& apt-get install -y git python3 python3-pip

RUN pip3 install uv

ADD scale25-crucible /opt/scale25-crucible

RUN uv pip install --system -r /opt/scale25-crucible/requirements.txt \
	&& uv pip install --system "git+https://github.com/trec-auto-judge/auto-judge-code.git#egg=trec_auto_judge[dspy]" \
	&& uv pip install --system --no-deps ir_datasets \
	&& python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords');"

ENV PYTHONPATH=/opt/scale25-crucible/ 

ADD umbrela_baseline.py /umbrela_baseline.py
