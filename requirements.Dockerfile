FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DIR "/vusic"

LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common && \
  add-apt-repository -y ppa:duggan/bats && \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-cffi sudo curl bats

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

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
