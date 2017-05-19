FROM ubuntu:16.04

MAINTAINER sword_carnon

COPY ./pip.conf /root/.pip/

RUN \
	#wget https://bootstrap.pypa.io/get-pip.py
	curl https://bootstrap.pypa.io/get-pip.py > get-pip.py

RUN \
	python get-pip.py

RUN \
	pip install sklearn \
	jieba \
	scipy

RUN \
	pip install gensim

COPY ./ /root/TextCluster/

WORKDIR /root/TextCluster/

CMD python TextCluster.py

