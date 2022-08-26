"""Evaluate trained anomaly detection models on pcap files."""
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns

from feature_extractor import pcap_to_dataframe, port_hierarchy_map_iot, preprocess_dataframe
from model_ae import Autoencoder


rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.markersize"] = 6
plot_width = 3.487  # in
plot_height = 2.155


def label_by_ip(df: pd.DataFrame, rules: List[Tuple[str, bool, str, bool, int]], default: int=-1) -> np.ndarray:
    """Label each packet based on a list of rules for IP addresses.

    Keyword arguments:
    df    -- DataFrame of extracted packet features.

    rules -- List of rules. Each rule is of type Tuple[str, bool, str,
    bool, int]. The first str, bool pair refers to the source IP
    address. The second str, bool pair refers to the destination IP
    address. The bools indicate whether the corresponding IP address
    should be included or excluded. The int is the label assigned to
    the packets that match the rule. Example: ("192.168.0.2", True,
    "192.168.0.10", True, 0) == if src addr IS 192.168.0.2 and dst
    addr IS 192.168.0.10 label as 0. ("192.168.0.2", True,
    "192.168.0.10", False, 1) == if src addr IS 192.168.0.2 and dst
    addr IS NOT 192.168.0.10 label as 1. You can refer to any IP
    address using an invalid IP address string and False.

    default -- default label assigned to packets that do not match the
    rules.
    """
    labels = np.full(df.shape[0], default)
    for srcip, srcinclude, dstip, dstinclude, label in rules:
        src_cmp = np.equal if srcinclude else np.not_equal
        dst_cmp = np.equal if dstinclude else np.not_equal
        mask = np.logical_and(src_cmp(df["ip_src"], srcip), dst_cmp(df["ip_dst"], dstip))
        labels[mask] = label

    return labels


def reconstruction_error(model, loss_function, samples):
    """Apply the model prediction to a list of samples."""
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error.numpy()


GLOBAL_MODEL_PATH = "xxx.tar"
VALID_NORMAL_DATA_PATH = "xxx.pickle"
VALID_ATTACK_DATA_PATH = "xxx.pickle"

# rules: see label_by_ip function.
rules: List[Tuple[str, bool, str, bool, int]] = []
# rules_map: mapping between a rule label and its description.
# Example: {0: "normal", 1: "Mirai C&C"}
rules_map: Dict[int, str] = {}
# text_info: put marks in the plot at certain timestamps.
# Example: (datetime(2022, 7, 13, 15, 5, 0), "start mirai bot")
text_info: List[Tuple[datetime, str]] = []

# === Load trained model ===
model = Autoencoder(69)
ckpt = torch.load(GLOBAL_MODEL_PATH)
model.load_state_dict(ckpt["state_dict"])
loss_func = F.mse_loss

# === dataset estimate threshold ===
df_raw_valid_normal = pd.read_pickle(VALID_NORMAL_DATA_PATH)
df_valid_normal = preprocess_dataframe(df_raw_valid_normal, port_mapping=port_hierarchy_map_iot)
timestamps_valid_normal = df_valid_normal["timestamp"].values
df_valid_normal = df_valid_normal.drop(columns=["timestamp"])
results_valid_normal = reconstruction_error(model, loss_func, torch.from_numpy(df_valid_normal.to_numpy(dtype=np.float32)))

fig, ax = plt.subplots()
ax.scatter(timestamps_valid_normal, results_valid_normal, linewidths=0, alpha=0.4)
ax.set_xlabel("timestamp")
ax.set_ylabel("MSE")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.show()

th = np.max(results_valid_normal[1:])
print(th)

# === dataset attack ===
df_raw_valid_attack = pd.read_pickle(VALID_ATTACK_DATA_PATH)

# manual labels
labels_valid_attack = label_by_ip(df_raw_valid_attack, rules, 10)

df_valid_attack = preprocess_dataframe(df_raw_valid_attack, port_mapping=port_hierarchy_map_iot)
timestamps_valid_attack = df_valid_attack["timestamp"].values
df_valid_attack = df_valid_attack.drop(columns=["timestamp"])
results_valid_attack = reconstruction_error(model, loss_func, torch.from_numpy(df_valid_attack.to_numpy(dtype=np.float32)))

results_df_valid_attack = pd.DataFrame({"ts": timestamps_valid_attack, "rec_err": results_valid_attack, "label": labels_valid_attack})
results_df_valid_attack["packet type"] = results_df_valid_attack["label"].map(rules_map)
fig, ax = plt.subplots()
sns.scatterplot(data=results_df_valid_attack, x="ts", y="rec_err", hue="packet type", linewidth=0, s=12, alpha=0.3, ax=ax, rasterized=True)
ax.axhline(y=th, linestyle=":", c="k")
if text_info:
    for info_ts, info_txt in text_info:
        ax.text(info_ts.timestamp(), results_valid_attack.max(), info_txt, ha="center", color="black", va="center", size=8, bbox=dict(boxstyle="circle,pad=0.1", lw=0.3, fc="white", ec="black"))
ax.set_xlabel("timestamp")
ax.set_ylabel("MSE")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.show()

# === metrics ===
labels_pred = (results_valid_attack > th * 1.05) + 0
labels_gnd_truth = (labels_valid_attack > 0) + 0
print(confusion_matrix(labels_gnd_truth, labels_pred, labels=[1, 0]))  # 0:normal, 1:attack; positive class is attack
tp, fn, fp, tn = confusion_matrix(labels_gnd_truth, labels_pred, labels=[1, 0]).ravel()
print("tp, fn, fp, tn = ", tp, ",", fn, ",", fp, ",", tn)
print("Accuracy: ", accuracy_score(labels_gnd_truth, labels_pred))
print("F1: ", f1_score(labels_gnd_truth, labels_pred, pos_label=1))
print("MCC: ", matthews_corrcoef(labels_gnd_truth, labels_pred))

fpr, tpr, thresholds = roc_curve(labels_gnd_truth, results_valid_attack, pos_label=1)
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()










# tests, ignore #
# only for vector plots
from scipy import stats
sec_since = timestamps - timestamps[0]

# reduce number of plot elements to control the output svg, pdf file size

# density
X = np.vstack([timestamps, results])
kde = stats.gaussian_kde(X)
Z = kde(X)

Z_s = (Z - np.min(Z)) / (np.max(Z) * 1.01 - np.min(Z))  # prevent probability to remove=1 on some points
Z_s[Z_s > 0.8] = 0.8
Z_s[(1700 < sec_since) & (sec_since < 5800) & (Z_s < 0.7)] = 0.05
Z_s[(labels == 1) & (results > 0.06)] = 0.85
Z_s[(labels == 1) & (results > 0.0299) & (results < 0.03019) & (4728.62 < sec_since) & (sec_since < 4728.84)] = 0.95
Z_s[(labels == 1) & (results > 0.0230211) & (results < 0.0231015) & (4728.62 < sec_since) & (sec_since < 4728.84)] = 0.95

plt.scatter(sec_since, Z_s)
plt.show()

mask = np.ones(sec_since.shape, dtype=bool)
for i in range(3):
    R = np.random.random(Z.shape)
    mask = np.logical_and(mask, (R - 0.0) > np.power(Z_s, 0.5))  # adjust threshold and power

# manual
# num_elems = results.shape[0]
# mask = np.zeros(results.shape)
# mask[:30000] = 1
# np.random.shuffle(mask)
# mask = mask.astype(bool)

# mask2 = (results < 9e-5)
# mask = np.logical_and(mask, mask2)
# mask = mask2

mask[0] = True
mask[-1] = True
sec_since_ = sec_since[mask]
results_ = results[mask]
labels_ = labels[mask]
print(np.sum(mask))
plt.close()
plt.scatter(sec_since_[labels_ == 1], results_[labels_ == 1], marker="^", c='#ff7f0e', alpha=0.4, s=50, linewidth=0, label="attack (ground truth)")
plt.scatter(sec_since_[labels_ == 0], results_[labels_ == 0], marker="o", c='#1f77b4', alpha=0.4, s=50, linewidth=0, label="normal (ground truth)")
plt.axhline(y=(th * 2.7), linewidth=0.5, c="k", label="threshold")

plt.hlines(y=0.05, xmin=1909, xmax=5590, colors="black", linestyles="dashed", linewidths=0.75)
plt.annotate(text="C&C active\nperiod", xy=(1850, 0.05), xytext=(0, 0.035), arrowprops=dict(facecolor='black', width=0.1, headwidth=1, headlength=1))

plt.text(2851, 0.049, "A", ha="center", color="white", va="center", size=6, bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="k", lw=0.1))
plt.text(3544, 0.049, "B", ha="center", color="white", va="center", size=6, bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="k", lw=0.1))
plt.text(4136, 0.049, "C", ha="center", color="white", va="center", size=6, bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="k", lw=0.1))
plt.text(4728, 0.049, "D", ha="center", color="white", va="center", size=6, bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="k", lw=0.1))

plt.xlabel("seconds since beginning of capture")
plt.ylabel("MSE")
plt.legend()
fig = plt.gcf()
fig.set_size_inches(plot_width, plot_height)
plt.tight_layout()
# plt.show()
fig.savefig(os.path.expanduser("~/packet_mse_mqtt2.pdf"), format="pdf")
plt.close()
