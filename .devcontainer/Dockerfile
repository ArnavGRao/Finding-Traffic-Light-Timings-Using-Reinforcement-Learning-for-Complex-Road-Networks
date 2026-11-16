FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg ca-certificates locales git curl \
    libx11-6 libgl1 libglu1-mesa \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository -y ppa:sumo/stable \
    && apt-get update && apt-get install -y --no-install-recommends \
    sumo sumo-tools sumo-doc \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt