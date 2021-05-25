from collections import namedtuple
from typing import Any, List, Dict, Optional
import configparser
import ipaddress
import json
import os
import time
import requests

PROJECT_NAME = "test_api"

Server = namedtuple("Server", ("addr", "port", "auth", "user", "password"))
Project = namedtuple("Project", ("name", "id", "grid_unit"))


def read_local_gns3_config():
    config = configparser.ConfigParser()
    with open(os.path.expanduser("~/.config/GNS3/2.2/gns3_server.conf")) as f:
        config.read_file(f)
    return config["Server"].get("host"), config["Server"].getint("port"), config["Server"].getboolean("auth"), config["Server"].get("user"), config["Server"].get("password")


def get_static_interface_config_file(iface: str, address: str, netmask: str, gateway: str) -> str:
    return (
        "# autogenerated\n"
        f"# Static config for {iface}\n"
        f"auto {iface}\n"
        f"iface {iface} inet static\n"
        f"\taddress {address}\n"
        f"\tnetmask {netmask}\n"
        f"\tgateway {gateway}\n"
        f"\tup echo nameserver {gateway} > /etc/resolv.conf\n"
    )


def template_id_from_name(template: List[Dict[str, Any]], name: str) -> Optional[str]:
    for d in template:
        if d["name"] == name:
            return d["template_id"]


def create_cluster_of_devices(server, project, num_devices, start_x, start_y, switch_template_id, device_template_id, start_ip):
    assert num_devices < 250  # test
    # create cluster switch
    payload = {"x": start_x + 5*project.grid_unit, "y": start_y - project.grid_unit}
    r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/templates/{switch_template_id}", data=json.dumps(payload), auth=(server.user, server.password))
    r.raise_for_status()
    switch_node_id = r.json()["node_id"]
    time.sleep(0.2)

    # create device grid
    devices_node_id = []
    dy = 0
    for i in range(num_devices):
        payload = {"x": start_x + (i%10)*project.grid_unit, "y": start_y + dy}
        r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/templates/{device_template_id}", data=json.dumps(payload), auth=(server.user, server.password))
        r.raise_for_status()
        devices_node_id.append(r.json()["node_id"])
        if i%10 == 9:
            dy += project.grid_unit
        time.sleep(0.2)
    assert len(devices_node_id) == num_devices

    # link devices to the switch
    devices_link_id = []
    for i, dev in enumerate(devices_node_id, start=1):
        payload = {"nodes": [{"adapter_number": 0, "node_id": dev, "port_number": 0},
                             {"adapter_number": 0, "node_id": switch_node_id, "port_number": i}]}
        r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links", data=json.dumps(payload), auth=(server.user, server.password))
        r.raise_for_status()
        devices_link_id.append(r.json()["link_id"])
        time.sleep(0.2)
    assert len(devices_link_id) == num_devices
    
    # change device configuration
    for i, dev in enumerate(devices_node_id, start=0):
        payload = get_static_interface_config_file("eth0", start_ip+i, "255.255.0.0", "192.168.0.1")
        r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/nodes/{dev}/files/etc/network/interfaces", data=payload, auth=(server.user, server.password))
        r.raise_for_status()
        print("Configured ", dev, " ", r.status_code)
        time.sleep(0.2)
    
    return {"switch_node_id": switch_node_id, "devices_node_id": devices_node_id, "devices_link_id": devices_link_id}


def start_capture(server, project, link_ids):
    for link in link_ids:
        r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links/{link}/start_capture", data={}, auth=(server.user, server.password))
        r.raise_for_status()
        result = r.json()
        print(f"Capturing {result['capturing']}, {result['capture_file_name']}")
        time.sleep(0.2)


def stop_capture(server, project, link_ids):
    for link in link_ids:
        r = requests.post(f"http://{server.addr}:{server.port}/v2/projects/{project.id}/links/{link}/stop_capture", data={}, auth=(server.user, server.password))
        r.raise_for_status()
        result = r.json()
        print(f"Capturing {result['capturing']}, {result['capture_file_name']}")
        time.sleep(0.2)


server = Server(*read_local_gns3_config())

req_version = requests.get(f"http://{server.addr}:{server.port}/v2/version", auth=(server.user, server.password))
req_version.raise_for_status()
print(req_version.json())

req_projects = requests.get(f"http://{server.addr}:{server.port}/v2/projects", auth=(server.user, server.password))
req_projects.raise_for_status()
projects = req_projects.json()
print(len(projects), " projects")
if projects:
    for p in projects:
        print(f"Name='{p['name']}', ID='{p['project_id']}'")

filtered_projects = list(filter(lambda x: x["name"]==PROJECT_NAME, projects))
if filtered_projects:
    p = filtered_projects[0]
    project = Project(name=p["name"], id=p["project_id"], grid_unit=int(p["grid_size"]*1.33))
    print(f"Project {PROJECT_NAME} exists. ", project)
else:
    # create the project
    # http://api.gns3.net/en/2.2/api/v2/controller/project/projects.html
    payload = {"name": PROJECT_NAME, "show_grid": True}
    r = requests.post(f"http://{server.addr}:{server.port}/v2/projects", data=json.dumps(payload), auth=(server.user, server.password))
    r.raise_for_status()
    project = Project(name=r.json()["name"], id=r.json()["project_id"], grid_unit=int(r.json()["grid_size"]*1.33))
    assert project.name == PROJECT_NAME
    print("Created project ", project)

# Por ahora crear los templates en GNS3 GUI

# "Ethernet switch 128"

# get templates
r = requests.get(f"http://{server.addr}:{server.port}/v2/templates", auth=(server.user, server.password))
r.raise_for_status()
templates = r.json()

# get template ids
sw_128_tem_id = template_id_from_name(templates, "Ethernet switch 128")
tem_id1 = template_id_from_name(templates, "iot-client")

