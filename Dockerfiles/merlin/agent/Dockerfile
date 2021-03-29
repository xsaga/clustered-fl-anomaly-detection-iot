FROM ubuntu:focal

RUN apt update && apt install -y --no-install-recommends \
    ca-certificates \
    wget \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/merlin
RUN wget https://github.com/Ne0nd0g/merlin/releases/latest/download/merlinAgent-Linux-x64.7z \
    && 7z x -pmerlin merlinAgent-Linux-x64.7z

FROM ubuntu:focal

RUN apt update && apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir paho-mqtt

COPY client.py /client.py
RUN chmod +x /client.py

COPY --from=0 /opt/merlin/merlinAgent-Linux-x64 /opt/merlin/merlinAgent-Linux-x64

CMD ["/bin/bash"]