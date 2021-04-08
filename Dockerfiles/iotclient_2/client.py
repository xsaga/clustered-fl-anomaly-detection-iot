#!/usr/bin/env python3

import os
import random
import time

import paho.mqtt.publish as publish

config = {"MQTT_BROKER_ADDR": "localhost",
          "MQTT_TOPIC_PUB": "test/topic",
          "SLEEP_TIME": 5,
          "SLEEP_TIME_SD": 0.5}

for key in config.keys():
    try:
        config[key] = os.environ[key]
    except KeyError:
        pass

config["MQTT_TOPIC_PUB"] = config["MQTT_TOPIC_PUB"] + "/" + os.environ["HOSTNAME"]

while True:
    publish.single(topic=config["MQTT_TOPIC_PUB"], payload=f"{random.gauss(10, 1):.2f}", hostname=config["MQTT_BROKER_ADDR"])
    sleep_time = random.gauss(float(config["SLEEP_TIME"]), float(config["SLEEP_TIME_SD"]))
    sleep_time = float(config["SLEEP_TIME"]) if sleep_time < 0 else sleep_time
    time.sleep(sleep_time)
