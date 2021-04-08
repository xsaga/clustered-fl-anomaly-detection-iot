#!/usr/bin/env python3

import os
import random
import sys
import shlex
import shutil
import subprocess
import time

config = {"COAP_ADDR_LIST": "",
          "SLEEP_TIME": 5,
          "SLEEP_TIME_SD": 0.1}

for key in config.keys():
    try:
        config[key] = os.environ[key]
    except KeyError:
        pass

config["COAP_ADDR_LIST"] = list(map(str.strip, config["COAP_ADDR_LIST"].split(";")))

coap_bin = shutil.which("coap-client")
if not coap_bin:
    sys.exit("No 'coap-client' binary found. Exiting.")

while True:
    for ip_addr in config["COAP_ADDR_LIST"]:
        resource = f"coap://{ip_addr}/time"
        cmd = f"{coap_bin} -m GET {resource}"

        # TODO include checks
        print(f"Sending: {cmd}")
        try:
            cmd_result = subprocess.run(shlex.split(cmd), capture_output=True, timeout=10, check=False)
        except subprocess.TimeoutExpired as e:
            print(e)
        print(f"Result stdout: {cmd_result.stdout}\nstderr: {cmd_result.stderr}")

    sleep_time = random.gauss(float(config["SLEEP_TIME"]), float(config["SLEEP_TIME_SD"]))
    sleep_time = float(config["SLEEP_TIME"]) if sleep_time < 0 else sleep_time
    time.sleep(sleep_time)
