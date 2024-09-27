# Anaconda Environment build
# docker build --platform linux/amd64 -t masters_env_py3_12 .
# docker tag 4d5a229daed4 trex22/masters_env_py3_12:latest
# docker push trex22/masters_env_py3_12:latest

FROM continuumio/miniconda3:main

# Set the working directory in the container to /app
WORKDIR /app

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# RUN echo "deb http://archive.debian.org/debian bullseye main" > /etc/apt/sources.list
RUN apt-get update && \
  apt-get install -y --no-install-recommends libdbus-glib-1-dev libgirepository1.0-dev libgtk-3-dev wget && \
  apt-get -y clean && apt-get -y autoremove && apt-get -y autoclean && \
  apt-get -y upgrade && apt-get -y clean && apt-get -y autoremove && apt-get -y autoclean

COPY Aug-2024-py312.yml .

RUN conda init bash

RUN CONDA_VERBOSITY=1 conda env create -f Aug-2024-py312.yml --name custom-environment && \
  conda clean --all -y && find /opt/conda/pkgs/ -delete && find /root/.cache/pip/ -delete

RUN echo "conda activate custom-environment" > ~/.bashrc
RUN echo "Make sure pytorch is installed:"
RUN python -c "import torch"

# sudo
# RUN apt-get -y install sudo && apt-get -y clean && apt-get -y autoremove && apt-get -y autoclean
# RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Cleanup
# RUN echo "conda clean --all" > ~/.bashrc
# RUN find ~/.cache/pip -delete
# RUN find /root/.cache/pip/ -delete
# RUN find /opt/conda/pkgs/ -delete
# RUN find /var/cache/ -delete

ENV PATH /opt/conda/envs/env/bin:$PATH
CMD ["/bin/bash"]
