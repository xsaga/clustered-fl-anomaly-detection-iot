from feature_extractor import pcap_to_dataframe, preprocess_dataframe, port_hierarchy_map_iot
from model_ae import Autoencoder
import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns


rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["lines.markersize"] = 6
plot_width = 3.487  # in
plot_height = 2.155


def label_by_ip(df :pd.DataFrame, ip_list):
    """if ip in src or dst, mark as anomalous."""
    labels = np.zeros(df.shape[0])
    labels[df["ip_src"].apply(lambda x: x in ip_list)] = 1
    labels[df["ip_dst"].apply(lambda x: x in ip_list)] = 1
    return labels


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
        mask = np.logical_and(src_cmp(df["ip_src"], srcip), dst_cmp(df["ip_dst"], dstip))
        labels[mask] = label

    return labels


def reconstruction_error(model, loss_function, samples):
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error


# trained model
model = Autoencoder(69)
ckpt = torch.load("global_model_round_30.tar")
model.load_state_dict(ckpt["state_dict"])
loss_func = F.mse_loss

rules = [('192.168.17.10', True, '192.168.1.1', True, 0),   # iot -> broker
         ('192.168.1.1', True, '192.168.17.10', True, 0),   # broker -> iot
         ('192.168.17.10', True, '192.168.0.2', True, 0),   # iot -> dns
         ('192.168.0.2', True, '192.168.17.10', True, 0),   # dns -> iot
         ('192.168.17.10', True, '192.168.0.3', True, 0),   # iot -> ntp
         ('192.168.0.3', True, '192.168.17.10', True, 0),   # ntp -> iot
         ('192.168.17.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.17.10', True, 1),
         ('192.168.17.10', True, '192.168.18.10', True, 2),
         ('192.168.18.10', True, '192.168.17.10', True, 2)]


# dataset estimate threshold
# df = pcap_to_dataframe("iot-client2-bot-1_normal.pcap", verbose=True)
df_raw = pd.read_pickle("iot-client2-bot-1_normal.pickle")
df = preprocess_dataframe(df_raw, port_mapping=port_hierarchy_map_iot)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])
results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()
results_df = pd.DataFrame({"ts":timestamps, "rec_err":results})
sns.scatterplot(data=results_df, x="ts", y="rec_err", linewidth=0, alpha=0.4)
plt.show()

th = np.max(results[1:])
print(th)

# dataset eval
# df = pcap_to_dataframe("iot-client2-bot-1_attack.pcap")
df_raw = pd.read_pickle("iot-client2-bot-1_attack.pickle")
# attack_victim_ip = ("192.168.0.254", "192.168.0.50")

## labels para los ataques
# labels = label_by_ip(df_raw, attack_victim_ip)
labels = label_by_ip_2(df_raw, rules, -1)

df = preprocess_dataframe(df_raw, port_mapping=port_hierarchy_map_iot)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])



results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))
results = results.numpy()

results_df = pd.DataFrame({"ts":timestamps, "rec_err":results, "label": labels})
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="label", linewidth=0, alpha=0.4)
plt.axhline(y=th, c="k")
plt.show()

labels_pred = (results>th)+0
confusion_matrix(labels, labels_pred, labels=[1,0])  # 0:normal, 1:attack; positive class is attack
tp, fn, fp, tn = confusion_matrix(labels, labels_pred, labels=[1,0]).ravel()
accuracy_score(labels, labels_pred)
f1_score(labels, labels_pred, pos_label=1)

fpr, tpr, thresholds = roc_curve(labels, results, pos_label=1)
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


## only for vector plots
from scipy import stats
sec_since = timestamps-timestamps[0]

# reduce number of plot elements to control the output svg, pdf file size

# density
X = np.vstack([timestamps, results])
kde = stats.gaussian_kde(X)
Z = kde(X)

Z_s = (Z - np.min(Z)) / (np.max(Z)*1.01 - np.min(Z))  # prevent probability to remove=1 on some points
Z_s[Z_s > 0.8] = 0.8
Z_s[(1700 < sec_since) & (sec_since < 5800) & (Z_s < 0.7)] = 0.05
Z_s[(labels==1) & (results > 0.06)] = 0.85
Z_s[(labels==1) & (results > 0.0299) & (results < 0.03019) & (4728.62 < sec_since) & (sec_since < 4728.84)] = 0.95
Z_s[(labels==1) & (results > 0.0230211) & (results < 0.0231015) & (4728.62 < sec_since) & (sec_since < 4728.84)] = 0.95

plt.scatter(sec_since, Z_s)
plt.show()

mask = np.ones(sec_since.shape, dtype=bool)
for i in range(3):
    R = np.random.random(Z.shape)
    mask = np.logical_and(mask, (R-0.0) > np.power(Z_s, 1/2))  # adjust threshold and power

# manual
# num_elems = results.shape[0]
# mask = np.zeros(results.shape)
# mask[:30000] = 1
# np.random.shuffle(mask)
# mask = mask.astype(bool)

#mask2 = (results < 9e-5)
#mask = np.logical_and(mask, mask2)
# mask = mask2

mask[0] = True
mask[-1] = True
sec_since_ = sec_since[mask]
results_ = results[mask]
labels_ = labels[mask]
print(np.sum(mask))
plt.close()
plt.scatter(sec_since_[labels_==1], results_[labels_==1], marker = "^", c='#ff7f0e', alpha=0.4, s=50, linewidth=0, label="attack (ground truth)")
plt.scatter(sec_since_[labels_==0], results_[labels_==0], marker = "o", c='#1f77b4', alpha=0.4, s=50, linewidth=0, label="normal (ground truth)")
plt.axhline(y=th*2.7, linewidth=0.5, c="k", label="threshold")

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
