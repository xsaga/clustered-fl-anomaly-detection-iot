# adapted from: https://github.com/ymirsky/Kitsune-py/blob/master/example.py

from Kitsune import Kitsune
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt


def label_by_ip_2(df, rules, default=-1):
    """
    example:
    rules = [("192.168.0.2", True, "192.168.0.10", True, 0),  # if src add IS 192.168.0.2 and dst addr IS 192.168.0.10 label as 0
             ("192.168.0.2", True, "192.168.0.10", False, 1)] # if src add IS 192.168.0.2 and dst addr IS NOT 192.168.0.10 label as 1
    """
    labels = np.full(df.shape[0], default)
    for srcip, srcinclude, dstip, dstinclude, label in rules:
        src_cmp = np.equal if srcinclude else np.not_equal
        dst_cmp = np.equal if dstinclude else np.not_equal
        mask = np.logical_and(src_cmp(df["ip.src"], srcip), dst_cmp(df["ip.dst"], dstip))
        labels[mask] = label

    return labels


# Load pcap
# The first XXX observations are clean...
path = "merged.pcap" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 443   # 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 2000  # 50000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

# Load tsv file into a DataFrame. Columns:
# frame.time_epoch, frame.len, eth.src,
# eth.dst, ip.src, ip.dst,
# tcp.srcport, tcp.dstport, udp.srcport,
# udp.dstport, icmp.type, icmp.code,
# arp.opcode, arp.src.hw_mac, arp.src.proto_ipv4,
# arp.dst.hw_mac, arp.dst.proto_ipv4, ipv6.src, ipv6.dst
df = pd.read_csv(path+".tsv", sep="\t")
total_packets = df.shape[0]
print("Total packets ", total_packets)

# Run Kitsune. 750k packets ~ 1 hour
print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    if i % 1000 == 0:
        print(i, " (", round(i/total_packets*100, 2), "%) elapsed: ", round(time.time() - start, 1), "s.")
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

labels = label_by_ip_2(df, rules, -1)
plt.scatter(list(range(1, total_packets+1)), RMSEs, s=0.5, c=labels, label=labels)
plt.legend()
plt.show()