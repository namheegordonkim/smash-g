FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3.5 python3-pip
RUN pip3 install tensorflow==1.3.0 numpy==1.13.0 scipy==0.17.1 matplotlib==1.5.3 pillow
RUN apt-get install -y python3-tk

RUN pip3 install tqdm
RUN pip3 install joblib
RUN pip3 install dill

RUN mkdir app
WORKDIR app