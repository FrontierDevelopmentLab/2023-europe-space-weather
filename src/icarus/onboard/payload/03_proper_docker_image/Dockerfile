FROM spacecloud.unibap.com/unibap/framework-missionimage

SHELL ["/bin/bash", "-c"]

RUN pip3.7 install astropy==4.3.1
RUN pip3.7 install compressai==1.2.4

COPY ./payload ./payload
COPY run_all.sh run_all.sh

RUN chmod +x run_all.sh

# One option - directly run python file
# WORKDIR /payload
# CMD ["python3.7","openvino_use_compressor.py"]

# Better option - run bash script
WORKDIR /
CMD ["bash","run_all.sh"]

# check image sizes with
#docker image ls
#docker image history helio_payload:latest
