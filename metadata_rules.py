"""rules, rules_map, text_info variables for the different attack scenarios."""
from datetime import datetime
from typing import Dict, List, Tuple


####################
# Cluster 0 (MQTT) #
####################

# Mirai scanload

rules = [('192.168.17.10', True, '192.168.1.1', True, 0),   # iot -> broker
         ('192.168.1.1', True, '192.168.17.10', True, 0),   # broker -> iot
         ('192.168.17.10', True, '192.168.0.2', True, 0),   # iot -> dns
         ('192.168.0.2', True, '192.168.17.10', True, 0),   # dns -> iot
         ('192.168.17.10', True, '192.168.0.3', True, 0),   # iot -> ntp
         ('192.168.0.3', True, '192.168.17.10', True, 0),   # ntp -> iot
         ('192.168.17.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.17.10', True, 2)]

rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 15, 5, 0), "A"), # start mirai bot
             (datetime(2022, 7, 13, 15, 17, 0), "B"), # loaded mirai in first victim
             (datetime(2022, 7, 13, 15, 26, 0), "C")] # loaded mirai in building-1

# Mirai DoS

rules = [('192.168.17.10', True, '192.168.1.1', True, 0),   # iot -> broker
         ('192.168.1.1', True, '192.168.17.10', True, 0),   # broker -> iot
         ('192.168.17.10', True, '192.168.0.2', True, 0),   # iot -> dns
         ('192.168.0.2', True, '192.168.17.10', True, 0),   # dns -> iot
         ('192.168.17.10', True, '192.168.0.3', True, 0),   # iot -> ntp
         ('192.168.0.3', True, '192.168.17.10', True, 0),   # ntp -> iot
         ('192.168.17.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.17.10', True, 2),
         ('192.168.17.10', True, '192.168.18.10', True, 3),
         ('192.168.18.10', True, '192.168.17.10', True, 3)]

# Also tag DNS attack:
timemask=np.logical_and(timestamps_valid_attack > 1657739000+300, timestamps_valid_attack < 1657739000+400)
dnsdstmask=df_raw_valid_attack["ip_dst"]=="192.168.0.2"
dnssrcmask=df_raw_valid_attack["ip_src"]=="192.168.0.2"
dnsmask=np.logical_or(dnsdstmask, dnssrcmask)
dnsattackmask=np.logical_and(timemask, dnsmask)
labels_valid_attack[dnsattackmask] = 3

rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             3: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 20, 55, 0), "A"), # start mirai
             (datetime(2022, 7, 13, 21, 6, 0), "B"), # primer ataque
             (datetime(2022, 7, 13, 21, 14, 0), "C"), # ultimo ataque
             (datetime(2022, 7, 13, 21, 20, 0), "D")] # stop mirai

# Merlin

rules = [('192.168.17.10', True, '192.168.1.1', True, 0),   # iot -> broker
         ('192.168.1.1', True, '192.168.17.10', True, 0),   # broker -> iot
         ('192.168.17.10', True, '192.168.0.2', True, 0),   # iot -> dns
         ('192.168.0.2', True, '192.168.17.10', True, 0),   # dns -> iot
         ('192.168.17.10', True, '192.168.0.3', True, 0),   # iot -> ntp
         ('192.168.0.3', True, '192.168.17.10', True, 0),   # ntp -> iot
         ('192.168.17.10', True, '192.168.34.10', True, 1),
         ('192.168.34.10', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.18.10', True, 2),
         ('192.168.18.10', True, '192.168.17.10', True, 2)]


rules_map = {0: "normal",
             1: "Merlin C&C",
             2: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 14, 41, 0), "A"), # start merlin agent
             (datetime(2022, 7, 18, 14, 54, 0), "B"), # upload hping3
             (datetime(2022, 7, 18, 15, 4, 0), "C"), # icmp
             (datetime(2022, 7, 18, 15, 9, 0), "D"), # udp
             (datetime(2022, 7, 18, 15, 13, 0), "E"), # syn
             (datetime(2022, 7, 18, 15, 17, 0), "F"), # ack
             (datetime(2022, 7, 18, 15, 25, 0), "G")] # stop merlin agent

# Masscan

rules = [('192.168.17.10', True, '192.168.1.1', True, 0),   # iot -> broker
         ('192.168.1.1', True, '192.168.17.10', True, 0),   # broker -> iot
         ('192.168.17.10', True, '192.168.0.2', True, 0),   # iot -> dns
         ('192.168.0.2', True, '192.168.17.10', True, 0),   # dns -> iot
         ('192.168.17.10', True, '192.168.0.3', True, 0),   # iot -> ntp
         ('192.168.0.3', True, '192.168.17.10', True, 0),   # ntp -> iot
         ('192.168.17.10', True, '192.168.35.10', True, 1),
         ('192.168.35.10', True, '192.168.17.10', True, 1)]


rules_map = {0: "normal",
             1: "Scanner",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 15, 44, 0), "A"), # scan 0.1kpps
             (datetime(2022, 7, 18, 15, 47, 0), "B"), # scan 1.0kpps
             (datetime(2022, 7, 18, 15, 55, 0), "C")] # scan 10kpps


####################
# Cluster 2 (CoAP) #
####################

# Mirai scanload

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.20.10', True, 2)]


rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 15, 5, 0), "A"), # start mirai bot
             (datetime(2022, 7, 13, 15, 17, 0), "B"), # loaded mirai in first victim
             (datetime(2022, 7, 13, 15, 23, 0), "C")] # loaded mirai in combined-cycle-1

# Mirai DoS

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.20.10', True, 2),
         ('192.168.20.10', True, '192.168.18.18', True, 3),
         ('192.168.18.18', True, '192.168.20.10', True, 3)]


rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             3: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 20, 30, 0), "A"), # start mirai
             (datetime(2022, 7, 13, 20, 40, 0), "B"), # primer ataque
             (datetime(2022, 7, 13, 20, 47, 0), "C"), # ultimo ataque
             (datetime(2022, 7, 13, 20, 54, 0), "D")] # stop mirai

# Merlin

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.34.10', True, 1),
         ('192.168.34.10', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.18.18', True, 2),
         ('192.168.18.18', True, '192.168.20.10', True, 2)]


rules_map = {0: "normal",
             1: "Merlin C&C",
             2: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 14, 40, 0), "A"), # start merlin agent
             (datetime(2022, 7, 18, 14, 52, 0), "B"), # upload hping3
             (datetime(2022, 7, 18, 15, 2, 0), "C"), # icmp
             (datetime(2022, 7, 18, 15, 7, 0), "D"), # udp
             (datetime(2022, 7, 18, 15, 12, 0), "E"), # syn
             (datetime(2022, 7, 18, 15, 16, 0), "F"), # ack
             (datetime(2022, 7, 18, 15, 25, 0), "G")] # stop merlin agent

# Masscan

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.35.10', True, 1),
         ('192.168.35.10', True, '192.168.20.10', True, 1)]


rules_map = {0: "normal",
             1: "Scanner",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 15, 43, 0), "A"), # scan 0.1kpps
             (datetime(2022, 7, 18, 15, 47, 0), "B"), # scan 1.0kpps
             (datetime(2022, 7, 18, 15, 55, 0), "C")] # scan 10kpps

# Nmap and coap amplification

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.35.10', True, 1),
         ('192.168.35.10', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.35.11', True, 2), # not in pcap because addr is spoofed
         ('192.168.35.11', True, '192.168.20.10', True, 2), # not in pcap because addr is spoofed
         ('192.168.20.10', True, '192.168.0.200', True, 3),
         ('192.168.0.200', True, '192.168.20.10', True, 3)]


rules_map = {0: "normal",
             1: "Scanner",
             2: "Amplification",  # not in pcap because addr is spoofed
             3: "Victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 20, 41, 0), "A"), # nmap random start
             (datetime(2022, 7, 18, 20, 54, 0), "B"), # nmap random end
             (datetime(2022, 7, 18, 21, 0, 0), "C"), # nmap p 5683
             (datetime(2022, 7, 18, 21, 2, 0), "D"), # nmap p 1000top
             (datetime(2022, 7, 18, 21, 20, 0), "E"), # nmap end
             (datetime(2022, 7, 18, 21, 25, 0), "F")] # coap ampli


###################
# Cluster 4 (cam) #
###################

# Mirai scanload

rules = [('192.168.17.15', True, '192.168.1.2', True, 0),
         ('192.168.1.2', True, '192.168.17.15', True, 0),
         ('192.168.17.15', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.17.15', True, 2)]


rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 15, 5, 0), "A"), # start mirai bot
             (datetime(2022, 7, 13, 15, 17, 0), "B")] # loaded mirai in first victim, this is museum-1

# Mirai DoS

rules = [('192.168.17.15', True, '192.168.1.2', True, 0),
         ('192.168.1.2', True, '192.168.17.15', True, 0),
         ('192.168.17.15', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.17.15', True, 2),
         ('192.168.17.15', True, '192.168.18.15', True, 3),
         ('192.168.18.15', True, '192.168.17.15', True, 3),]


rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             3: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 13, 18, 57, 0), "A"), # start mirai
             (datetime(2022, 7, 13, 19, 10, 0), "B"), # primer ataque
             (datetime(2022, 7, 13, 19, 17, 0), "C"), # ultimo ataque
             (datetime(2022, 7, 13, 19, 20, 0), "D")] # stop mirai

# Merlin

rules = [('192.168.17.15', True, '192.168.1.2', True, 0),
         ('192.168.1.2', True, '192.168.17.15', True, 0),
         ('192.168.17.15', True, '192.168.34.10', True, 1),
         ('192.168.34.10', True, '192.168.17.15', True, 1),
         ('192.168.17.15', True, '192.168.18.15', True, 2),
         ('192.168.18.15', True, '192.168.17.15', True, 2),]


rules_map = {0: "normal",
             1: "Merlin C&C",
             2: "DoS victim",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 14, 40, 0), "A"), # start merlin agent
             (datetime(2022, 7, 18, 14, 50, 0), "B"), # upload hping3
             (datetime(2022, 7, 18, 15, 1, 0), "C"), # icmp
             (datetime(2022, 7, 18, 15, 6, 0), "D"), # udp
             (datetime(2022, 7, 18, 15, 10, 0), "E"), # syn
             (datetime(2022, 7, 18, 15, 14, 0), "F"), # ack
             (datetime(2022, 7, 18, 15, 25, 0), "G")] # stop merlin agent

# Masscan

rules = [('192.168.17.15', True, '192.168.1.2', True, 0),
         ('192.168.1.2', True, '192.168.17.15', True, 0),
         ('192.168.17.15', True, '192.168.35.10', True, 1),
         ('192.168.35.10', True, '192.168.17.15', True, 1)]


rules_map = {0: "normal",
             1: "Scanner",
             10: "Others"}

text_info = [(datetime(2022, 7, 18, 15, 43, 0), "A"), # scan 0.1kpps
             (datetime(2022, 7, 18, 15, 47, 0), "B"), # scan 1.0kpps
             (datetime(2022, 7, 18, 15, 55, 0), "C")] # scan 10kpps
