import math
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from scapy.all import ARP, ICMP, IP, IPv6, PcapReader, TCP, UDP, raw
from tqdm import tqdm

port_basic_three_map: List[Tuple[Sequence[int], str]] = [
    (range(0, 1024), "system"),
    (range(1024, 49152), "user"),
    (range(49152, 65536), "dynamic")
]

port_hierarchy_map: List[Tuple[Sequence[int], str]] = [
    ([80, 280, 443, 591, 593, 777, 488, 1183, 1184, 2069, 2301, 2381, 8008, 8080], "httpPorts"),
    ([24, 25, 50, 58, 61, 109, 110, 143, 158, 174, 209, 220, 406, 512, 585, 993, 995], "mailPorts"),
    ([42, 53, 81, 101, 105, 261], "dnsPorts"),
    ([20, 21, 47, 69, 115, 152, 189, 349, 574, 662, 989, 990], "ftpPorts"),
    ([22, 23, 59, 87, 89, 107, 211, 221, 222, 513, 614, 759, 992], "shellPorts"),
    ([512, 514], "remoteExecPorts"),
    ([13, 56, 113, 316, 353, 370, 749, 750], "authPorts"),
    ([229, 464, 586, 774], "passwordPorts"),
    ([114, 119, 532, 563], "newsPorts"),
    ([194, 258, 531, 994], "chatPorts"),
    ([35, 92, 170, 515, 631], "printPorts"),
    ([13, 37, 52, 123, 519, 525], "timePorts"),
    ([65, 66, 118, 150, 156, 217], "dbmsPorts"),
    ([546, 547, 647, 847], "dhcpPorts"),
    ([43, 63], "whoisPorts"),
    (range(137, 139 + 1), "netbiosPorts"),
    ([88, 748, 750], "kerberosPorts"),
    ([111, 121, 369, 530, 567, 593, 602], "RPCPorts"),
    ([161, 162, 391], "snmpPorts"),
    (range(0, 1024), "PRIVILEGED_PORTS"),
    (range(1024, 65536), "NONPRIVILEGED_PORTS")
]

port_hierarchy_map_iot: List[Tuple[Sequence[int], str]] = [
    ([1883, 8883], "mqttPorts"),
    ([5683, 5684], "coapPorts"),
    ([8554, 8322, 8000, 8001, 8002, 8003, 1935, 8888], "rtspPorts"),
    ([80, 280, 443, 591, 593, 777, 488, 1183, 1184, 2069, 2301, 2381, 8008, 8080], "httpPorts"),
    ([24, 25, 50, 58, 61, 109, 110, 143, 158, 174, 209, 220, 406, 512, 585, 993, 995], "mailPorts"),
    ([42, 53, 81, 101, 105, 261], "dnsPorts"),
    ([20, 21, 47, 69, 115, 152, 189, 349, 574, 662, 989, 990], "ftpPorts"),
    ([22, 23, 59, 87, 89, 107, 211, 221, 222, 513, 614, 759, 992], "shellPorts"),
    ([512, 514], "remoteExecPorts"),
    ([13, 56, 113, 316, 353, 370, 749, 750], "authPorts"),
    ([229, 464, 586, 774], "passwordPorts"),
    ([114, 119, 532, 563], "newsPorts"),
    ([194, 258, 531, 994], "chatPorts"),
    ([35, 92, 170, 515, 631], "printPorts"),
    ([13, 37, 52, 123, 519, 525], "timePorts"),
    ([65, 66, 118, 150, 156, 217], "dbmsPorts"),
    ([546, 547, 647, 847], "dhcpPorts"),
    ([43, 63], "whoisPorts"),
    (range(137, 139 + 1), "netbiosPorts"),
    ([88, 748, 750], "kerberosPorts"),
    ([111, 121, 369, 530, 567, 593, 602], "RPCPorts"),
    ([161, 162, 391], "snmpPorts"),
    (range(0, 1024), "PRIVILEGED_PORTS"),
    (range(1024, 65536), "NONPRIVILEGED_PORTS")
]

IP_PROTOCOL_CATEGORIES = ["TCP", "UDP", "ICMP"]
ip_protocol_dtype = CategoricalDtype(categories=IP_PROTOCOL_CATEGORIES)


def entropy(x: bytearray) -> float:
    cnt = np.bincount(x, minlength=256)
    return stats.entropy(cnt, base=2)


def ip_flag_to_str(flag: int) -> str:
    return str(IP(flags=flag).flags)


def tcp_flag_to_str(flag: int) -> str:
    return str(TCP(flags=flag).flags)


def port_to_categories(port_map: List[Tuple[Sequence[int], str]], port: int) -> str:
    for p_range, p_name in port_map:
        if port in p_range:
            return p_name

    return ""


def get_pcap_packet_count(filename: str) -> Optional[int]:
    if os.name == "nt":
        extrapath = os.environ["PATH"] + os.pathsep + "C:\\Program Files\\Wireshark"
    else:
        extrapath = None

    capinfos_bin = shutil.which("capinfos", path=extrapath)
    if not capinfos_bin:
        return None

    try:
        capinfos_out = subprocess.run([capinfos_bin, "-M", "-c", filename], check=True, capture_output=True, encoding="utf-8")
        packet_count = int(capinfos_out.stdout.strip().split()[-1])
    except subprocess.CalledProcessError as e:
        print(e)
        return None

    return packet_count


def pcap_to_dataframe(pcap_filename: str, verbose=False) -> pd.DataFrame:
    # count number of packets
    packet_count = get_pcap_packet_count(pcap_filename)
    if verbose and packet_count:
        print(f"Number of packets in capture {packet_count}")

    # rdpcap uses too much memory
    # capture = rdpcap(pcap_filename)
    pkt_features = []

    with PcapReader(pcap_filename) as capture:
        for pkt in tqdm(capture, total=packet_count, disable=(not verbose)):
            features = {"timestamp": 0,
                        "packet_length": 0,
                        "iat": 0,
                        "h": 0,
                        "ip_src": "",
                        "ip_dst": "",
                        "ip_tos": 0,
                        "ip_flags": 0,
                        "ip_ttl": 0,
                        "ip_protocol": "",
                        "sport": 0,
                        "dport": 0,
                        "tcp_flags": 0,
                        "window": 0}

            # filtering
            if IPv6 in pkt:
                continue
            if ARP in pkt:
                continue

            features["timestamp"] = float(pkt.time)

            features["packet_length"] = len(pkt)

            try:
                features["iat"] = features["timestamp"] - pkt_features[-1]["timestamp"]
            except IndexError:
                features["iat"] = 0.0

            features["h"] = entropy(bytearray(raw(pkt)))

            if pkt.haslayer(IP):
                lyr = pkt.getlayer(IP)
                features["ip_src"] = lyr.src
                features["ip_dst"] = lyr.dst
                features["ip_tos"] = lyr.tos
                features["ip_flags"] = lyr.flags.value
                features["ip_ttl"] = lyr.ttl

                if pkt.haslayer(TCP):
                    lyr = pkt.getlayer(TCP)
                    features["ip_protocol"] = "TCP"
                    features["sport"] = lyr.sport
                    features["dport"] = lyr.dport
                    features["tcp_flags"] = lyr.flags.value
                    features["window"] = lyr.window
                elif pkt.haslayer(UDP):
                    lyr = pkt.getlayer(UDP)
                    features["ip_protocol"] = "UDP"
                    features["sport"] = lyr.sport
                    features["dport"] = lyr.dport
                elif pkt.haslayer(ICMP):
                    lyr = pkt.getlayer(ICMP)
                    features["ip_protocol"] = "ICMP"

            pkt_features.append(features)

    return pd.DataFrame(pkt_features)


def preprocess_dataframe(input_df: pd.DataFrame, port_mapping: Optional[List[Tuple[Sequence[int], str]]]=None, sport_bins: Optional[List[int]]=None, dport_bins: Optional[List[int]]=None) -> pd.DataFrame:
    """
    Ex:
    sport_bins = [0, 1883, 35783, 40629, 45765, 51039, 56023, 65536]  # from federated binning extractor (default: [0, 1024, 49152, 65536])
    dport_bins = [0, 1883, 35690, 41810, 48285, 54791, 65536]  # from federated binning extractor (default: [0, 1024, 49152, 65536])
    """

    if (port_mapping is None) and (sport_bins is None) and (dport_bins is None):
        port_mapping = port_basic_three_map
        print("Using default port mapping: ", port_mapping)

    df = input_df.copy()

    if port_mapping:
        # one hot encoding for sport and dport using port mapping
        port_categories = list(map(lambda x: x[1], port_mapping))
        port_dtype = CategoricalDtype(categories=port_categories)
        df["sport"] = df["sport"].apply(lambda p: port_to_categories(port_mapping, p)).astype(port_dtype)
        df["dport"] = df["dport"].apply(lambda p: port_to_categories(port_mapping, p)).astype(port_dtype)
    else:
        # one hot encoding for sport and dport using port bins
        df["sport"] = pd.cut(df["sport"], bins=sport_bins)
        df["dport"] = pd.cut(df["dport"], bins=dport_bins)

    sport_onehot = pd.get_dummies(df["sport"], prefix="sport")
    dport_onehot = pd.get_dummies(df["dport"], prefix="dport")

    # one hot encoding for ip_protocol
    df["ip_protocol"] = df["ip_protocol"].astype(ip_protocol_dtype)
    ip_protocol_onehot = pd.get_dummies(df["ip_protocol"], prefix="ip_proto")

    # binary encoding for the flags
    # IP.flags.size = 3
    # TCP.flags.size = 9
    ip_flags = df["ip_flags"].values.reshape((-1, 1)).astype(np.uint8)
    ip_flags = np.unpackbits(ip_flags, axis=1, bitorder="little")[:, :IP.flags.size]
    ip_flags_df = pd.DataFrame(ip_flags, columns=[f"ip_flag_{x}" for x in IP.flags.names])

    tcp_flags = df["tcp_flags"].values.reshape((-1, 1)).astype(np.uint16).view(np.uint8)
    tcp_flags = np.unpackbits(tcp_flags, axis=1, bitorder="little")[:, :TCP.flags.size]
    tcp_flags_df = pd.DataFrame(tcp_flags, columns=[f"tcp_flag_{x}" for x in TCP.flags.names])

    # scaling
    df["packet_length"] = df["packet_length"] / 1514
    df["iat"] = np.log1p(df["iat"])
    df["window"] = df["window"] / 65535
    df["ip_ttl"] = df["ip_ttl"] / 255
    df["ip_tos"] = df["ip_tos"] / 255
    df["h"] = df["h"] / (-math.log2(1 / 256))  # entropy/8.0

    # drop features
    df = df.drop(columns=["ip_src", "ip_dst", "ip_flags", "ip_protocol", "tcp_flags", "sport", "dport"])

    df = df.join([ip_protocol_onehot, sport_onehot, dport_onehot, ip_flags_df, tcp_flags_df])

    return df
