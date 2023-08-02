FROM nvcr.io/nvidia/pytorch:23.07-py3

USER root
RUN mkdir /icarus
WORKDIR /icarus

# install python dependencies
ADD ./2023-europe-space-weather /icarus

RUN apt install -y libjpeg-dev libpng-dev

# copy code
RUN pip install --upgrade --force-reinstall torch
RUN pip install -e .[ground]

ENTRYPOINT ["python", "-m", "sunerf.sunerf"]
