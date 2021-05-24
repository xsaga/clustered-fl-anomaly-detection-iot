import math
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from scapy.all import *


PORT_CATEGORIES = ["system", "user", "dynamic"]
IP_PROTOCOL_CATEGORIES = ["TCP", "UDP", "ICMP"]
port_dtype = CategoricalDtype(categories=PORT_CATEGORIES)
ip_protocol_dtype = CategoricalDtype(categories=IP_PROTOCOL_CATEGORIES)


def entropy(x: bytearray) -> float:
    cnt = np.bincount(x, minlength=256)
    return stats.entropy(cnt, base=2)


def ip_flag_to_str(flag: int) -> str:
    return str(IP(flags=flag).flags)


def tcp_flag_to_str(flag: int) -> str:
    return str(TCP(flags=flag).flags)


def port_to_categories(port: int) -> str:  #! incrementar la clasificacion, jerarquias mas detalladas.
    if port in range(1, 1024):
        return "system"
    elif port in range(1024, 49152):
        return "user"
    elif port in range(49152, 65536):
        return "dynamic"
    return ""


def pcap_to_dataframe(pcap_filename: str) -> pd.DataFrame:
    capture = rdpcap(pcap_filename)
    pkt_features = []

    for pkt in capture:
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


def preprocess_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    # one hot encoding for sport and dport
    df["sport"] = df["sport"].apply(port_to_categories).astype(port_dtype)
    df["dport"] = df["dport"].apply(port_to_categories).astype(port_dtype)
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
    # df["packet_length"] = np.log1p(df["packet_length"])
    # df["iat"] = np.log1p(df["iat"])
    # df["window"] = np.log1p(df["window"])
    # df["ip_ttl"] = np.log1p(df["ip_ttl"])
    # df["ip_tos"] = np.log1p(df["ip_tos"])
    # df["h"] = df["h"]/(-math.log2(1/256))  # entropy/8.0

    df["packet_length"] = df["packet_length"]/1514
    df["iat"] = np.log1p(df["iat"])
    df["window"] = df["window"]/65535
    df["ip_ttl"] = df["ip_ttl"]/255
    df["ip_tos"] = df["ip_tos"]/255
    df["h"] = df["h"]/(-math.log2(1/256))  # entropy/8.0


    # drop features
    df = df.drop(columns=["ip_src", "ip_dst", "ip_flags", "ip_protocol", "tcp_flags", "sport", "dport"])

    df = df.join([ip_protocol_onehot, sport_onehot, dport_onehot, ip_flags_df, tcp_flags_df])

    return df
