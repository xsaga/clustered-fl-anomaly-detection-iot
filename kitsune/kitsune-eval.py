import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, matthews_corrcoef


def label_by_ip(df, rules, default=-1):
    """
    example:
    rules = [("192.168.0.2", True, "192.168.0.10", True, 0),  # if src add IS 192.168.0.2 and dst addr IS 192.168.0.10 label as 0
             ("192.168.0.2", True, "192.168.0.10", False, 1)] # if src add IS 192.168.0.2 and dst addr IS NOT 192.168.0.10 label as 1
    """
    labels = np.full(df.shape[0], default)
    # set to normal: ipv6, arp. To make comparison with FL IDS.
    labels[df["ipv6.src"].notna()] = 0
    labels[df["ipv6.dst"].notna()] = 0
    labels[df["arp.opcode"].notna()] = 0
    for srcip, srcinclude, dstip, dstinclude, label in rules:
        src_cmp = np.equal if srcinclude else np.not_equal
        dst_cmp = np.equal if dstinclude else np.not_equal
        mask = np.logical_and(src_cmp(df["ip.src"], srcip), dst_cmp(df["ip.dst"], dstip))
        labels[mask] = label

    return labels


rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.markersize"] = 6
plot_width = 3.487  # in
plot_height = 2.155

# Load serialized results
path = "merged.pcap"
path_tsv_file = path + ".tsv"
path_serialized_rmse = path + ".rmse.npy"

# metadata
rules = []
rules_map = {}
text_info = []

# Load tsv file into a DataFrame. Columns:
# frame.time_epoch, frame.len, eth.src,
# eth.dst, ip.src, ip.dst,
# tcp.srcport, tcp.dstport, udp.srcport,
# udp.dstport, icmp.type, icmp.code,
# arp.opcode, arp.src.hw_mac, arp.src.proto_ipv4,
# arp.dst.hw_mac, arp.dst.proto_ipv4, ipv6.src, ipv6.dst
df = pd.read_csv(path_tsv_file, sep="\t")
total_packets = df.shape[0]
print("Total packets ", total_packets)

# Load RMSEs
RMSEs = np.load(path_serialized_rmse)
assert RMSEs.shape[0] == df.shape[0]

labels = label_by_ip(df, rules, 10)
first_nonzero = np.nonzero(RMSEs)[0][0]
th = np.max(RMSEs[first_nonzero:first_nonzero+100])

rmse_df = pd.DataFrame({"ts": df["frame.time_epoch"][first_nonzero:], "RMSE": RMSEs[first_nonzero:], "label": labels[first_nonzero:]})
rmse_df["packet type"] = rmse_df["label"].map(rules_map)

fig, ax = plt.subplots()
sns.scatterplot(data=rmse_df, x="ts", y="RMSE", hue="packet type", s=4, linewidth=0, ax=ax, rasterized=True)
ax.axhline(y=th, linestyle=":", c="k")
if text_info:
    for info_ts, info_txt in text_info:
        ax.text(info_ts.timestamp(), RMSEs.max(), info_txt, ha="center", color="black", va="center", size=8, bbox=dict(boxstyle="circle,pad=0.1", lw=0.3, fc="white", ec="black"), zorder=99)
ax.set_yscale("log")
ax.set_xlabel("timestamp")
ax.set_ylabel("RMSE")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.show()


# === metrics ===
RMSEs_red = RMSEs[first_nonzero:]
labels_red = labels[first_nonzero:]
labels_pred = (RMSEs_red > th*1.05)+0
labels_gnd_truth = (labels_red > 0)+0
print(confusion_matrix(labels_gnd_truth, labels_pred, labels=[1,0]))  # 0:normal, 1:attack; positive class is attack
tp, fn, fp, tn = confusion_matrix(labels_gnd_truth, labels_pred, labels=[1,0]).ravel()
print(f"TP, FN, FP, TN = {tp}, {fn}, {fp}, {tn}")
print("Accuracy: ", accuracy_score(labels_gnd_truth, labels_pred))
print("F1: ", f1_score(labels_gnd_truth, labels_pred, pos_label=1))
print("MCC: ", matthews_corrcoef(labels_gnd_truth, labels_pred))
