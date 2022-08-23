# adapted from: https://github.com/ymirsky/Kitsune-py/blob/master/example.py

from Kitsune import Kitsune
import numpy as np
import time
import pandas as pd


# Load pcap
# The first XXX observations are clean...
# the pcap, pcapng, or tsv file to process.
path = "merged.pcap"
# the number of packets to process
packet_limit = np.Inf

# KitNET params:
maxAE = 10  # maximum size for any autoencoder in the ensemble layer
# 5000 # the number of instances taken to learn the feature mapping (the ensemble's architecture)
FMgrace = 244
# 50000 # the number of instances used to train the anomaly detector (ensemble itself)
ADgrace = 2443 - FMgrace

# Build Kitsune
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

# Load tsv file into a DataFrame. Columns:
# frame.time_epoch, frame.len, eth.src,
# eth.dst, ip.src, ip.dst,
# tcp.srcport, tcp.dstport, udp.srcport,
# udp.dstport, icmp.type, icmp.code,
# arp.opcode, arp.src.hw_mac, arp.src.proto_ipv4,
# arp.dst.hw_mac, arp.dst.proto_ipv4, ipv6.src, ipv6.dst
df = pd.read_csv(path + ".tsv", sep="\t")
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
    i += 1
    if i % 1000 == 0:
        print(i, " (", round(i / total_packets * 100, 2), "%) elapsed: ", round(time.time() - start, 1), "s.")
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

# serialize results
print("Serializing RMSEs results")
np.save(path + ".rmse.npy", RMSEs)
