FROM nvcr.io/nvidia/pytorch:23.07-py3

USER root
RUN mkdir /icarus
WORKDIR /icarus

# install python dependencies
ADD ./2023-europe-space-weather /icarus

# copy code
RUN pip install -e .[ground]

CMD [ "/bin/bash" ]
