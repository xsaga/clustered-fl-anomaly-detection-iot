import json
import re
import random

with open("iot_anomaly_detection_2.gns3", "r") as f:
    project = json.loads(f.read())

width = project["scene_width"]
height = project["scene_height"]

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'


topology = project["topology"]

for drawing in topology["drawings"]:
    x = drawing["x"] + width//2
    y = drawing["y"] + height//2
    try:
        text = re.findall(r"(<text.*</text>).*", drawing["svg"], flags=re.DOTALL)[0]
        text = text.replace("<text", f'<text x="{x}" y="{y}"') + "\n"
    except Exception:  # yes
        print("!")
        continue
    svg += text

node_coords = dict()
for node in topology["nodes"]:
    x = node["x"] + width//2
    y = node["y"] + height//2
    node_coords[node["node_id"]] = (x, y)
    svg += f'<circle cx="{x}" cy="{y}" r="40" stroke="black" stroke-width="3" fill="red" />\n'

    label = node["label"]
    lx = label["x"] # relative to x
    ly = label["y"] # relative to y
    lt = label["text"]
    svg += f'<text x="{x+lx}" y="{y+ly}" fill="blue">{lt}</text>\n'


for link in topology["links"]:
    l1, l2 = link["nodes"]    
    l1x, l1y = node_coords[l1["node_id"]]
    l2x, l2y = node_coords[l2["node_id"]]
    svg += f'<line x1="{l1x}" y1="{l1y}" x2="{l2x}" y2="{l2y}" style="stroke:rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)});stroke-width:10"/>\n'

svg += '</svg>\n'
with open("gns3_topology.svg", "w") as of:
    of.write(svg)
