FROM ubuntu:18.04

ENV DIR "/vusic"

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository -y ppa:duggan/bats && \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-cffi sudo curl bats

RUN ln -s /usr/bin/python3 /usr/bin/python && \
  ln -s /usr/bin/pip3 /usr/bin/pip

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR ${DIR}
COPY . ${DIR}

COPY environment.sh .
RUN ["chmod", "+x", "environment.sh"]
RUN ./environment.sh

CMD ["bats", "requirements_test.bats"]
