import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns
from feature_extractor import pcap_to_dataframe


pcaps = [f for f in Path("./").iterdir() if f.is_file() and f.match("*.pcap")]

list_dport = []
list_sport = []

for i, f in enumerate(pcaps):
    print(f"{i}/{len(pcaps)}: {f.name}")
    df = pcap_to_dataframe(f.name)
    list_dport.append(df["dport"])
    list_sport.append(df["sport"])

with open("list_dport_sport.pickle", "wb") as f:
    pickle.dump({"list_dport":list_dport, "list_sport":list_sport}, f)

# with open("list_dport_sport.pickle", "rb") as f:
#     loaded = pickle.load(f)
#     list_dport = loaded["list_dport"]
#     list_sport = loaded["list_sport"]

########################################## merge local bins ### (no sale tan bien)
list_dport_bins = [KBinsDiscretizer(n_bins=10, strategy="quantile").fit(l.values.reshape(-1,1)).bin_edges_[0] for l in list_dport]
list_sport_bins = [KBinsDiscretizer(n_bins=10, strategy="quantile").fit(l.values.reshape(-1,1)).bin_edges_[0] for l in list_sport]

bins_all_dport = np.concatenate(list_dport_bins).astype(int)
bins_all_sport = np.concatenate(list_sport_bins).astype(int)
# filter here?
bins_dport_uniq = np.unique(np.sort(bins_all_dport))
bins_sport_uniq = np.unique(np.sort(bins_all_sport))

########################################## global discretize ### (mejor)

list_dport_all = pd.concat(list_dport)
list_sport_all = pd.concat(list_sport)

dport_bins = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(list_dport_all.values.reshape(-1,1)).bin_edges_[0].astype(np.int32)
sport_bins = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(list_sport_all.values.reshape(-1,1)).bin_edges_[0].astype(np.int32)

if dport_bins[0] > 0:
    dport_bins = np.insert(dport_bins, 0, 0)
dport_bins[-1] = 2**16

if sport_bins[0] > 0:
    sport_bins = np.insert(sport_bins, 0, 0)
sport_bins[-1] = 2**16

pd.cut(list_dport_all.values, bins=dport_bins).value_counts().plot(kind="bar", rot=40); plt.tight_layout(); plt.show()
pd.cut(list_sport_all.values, bins=sport_bins).value_counts().plot(kind="bar", rot=40); plt.tight_layout(); plt.show()

df_ports = pd.DataFrame({"destination port": pd.cut(list_dport_all.values, bins=dport_bins),
                         "source port": pd.cut(list_sport_all.values, bins=sport_bins)})

rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
plot_width = 3.487  # in
plot_height = 2.615

fig, axes = plt.subplots(nrows=2, ncols=1)
df_ports["source port"].value_counts(normalize=True, sort=False).plot(kind="bar", ax=axes[0], rot=15)
axes[0].legend()
df_ports["destination port"].value_counts(normalize=True, sort=False).plot(kind="bar", ax=axes[1], rot=15)
axes[1].legend()
axes[1].set_xlabel("port bins")
axes[1].set_ylabel("normalized count")
fig.set_size_inches(plot_width, plot_height)
plt.tight_layout()
fig.savefig(os.path.expanduser("~/src_dst_port_binning.pdf"), format="pdf")
# plt.show()

############################################################################## (tests, no usar)
list_bins = []

with open("results_binning_quantile.txt", "w") as o:
    for f in pcaps:
        print(f.name)
        df = pcap_to_dataframe(f.name)
        discr = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(df["dport"].values.reshape(-1,1))
        print(discr.bin_edges_)
        list_bins.append(discr.bin_edges_)
        o.write(f"{f.name}: {discr.bin_edges_}\n")
        print("------------\n")


bins_all_array = np.concatenate(list_bins).astype(int)
v,c = np.unique(bins_all_array[bins_all_array>1024], return_counts=True)
bins_gt1024_selected = v[c>2]

selected_bins = np.concatenate([bins_all_array[bins_all_array<=1024], bins_gt1024_selected])

# bins_uniq = np.unique(np.sort(bins_all_array))
bins_uniq = np.unique(np.sort(selected_bins))

# igual no hace falta el astype int
bins_centered = np.mean(np.vstack([bins_uniq[:-1], bins_uniq[1:]]), axis=0)

# y poner 0 al principio de los bins y np.inf (o 2**16) al final
# pd.cut(x, bins=bins)
# pd.get_dummies()
bins=discr.bin_edges_[0]

bins=np.insert(bins, 0, -1)
bins=np.append(bins, 2**16)
pd.cut(df["dport"].values, bins=bins).value_counts().plot(kind="bar", rot=0)



In [329]: list_cuts = []
     ...: with open("results_binning_quantile.txt", "w") as o:
     ...:     for i,f in enumerate(pcaps):
     ...:         print(i, f.name)
     ...:         df = pcap_to_dataframe(f.name)
     ...:         dport = df["dport"]
     ...:         list_cuts.append(pd.cut(dport.values, bins=bins))

list_cuts_series = [ pd.Series(x) for x in list_cuts ]
todos = pd.concat(list_cuts_series)
todos.value_counts(sort=False).plot(kind="bar", rot=60); plt.show()


# #################### vvvv
In [386]: list_dport = []
          list_sport = []
     ...: for i,f in enumerate(pcaps):
     ...:     print(i, f.name)
     ...:     df = pcap_to_dataframe(f.name)
     ...:     dport = df["dport"]
              sport = df["sport"]
     ...:     list_dport.append(dport)
              list_sport.append(sport)

list_dport_count_raw = [x.value_counts(normalize=False).sort_index() for x in list_dport]
list_dport_count_norm = [x.value_counts(normalize=True).sort_index() for x in list_dport]

list_sport_count_raw = [x.value_counts(normalize=False).sort_index() for x in list_sport]
list_sport_count_norm = [x.value_counts(normalize=True).sort_index() for x in list_sport]

todos_dport_count_raw = list_dport_count_raw[0]
for ps in list_dport_count_raw[1:]:
    todos_dport_count_raw = todos_dport_count_raw.add(ps, fill_value=0)

todos_sport_count_raw = list_sport_count_raw[0]
for ps in list_sport_count_raw[1:]:
    todos_sport_count_raw = todos_sport_count_raw.add(ps, fill_value=0)

todos_dport_count_norm = list_dport_count_norm[0]
for ps in list_dport_count_norm[1:]:
    todos_dport_count_norm = todos_dport_count_norm.add(ps, fill_value=0)
todos_dport_count_norm = todos_dport_count_norm/todos_dport_count_norm.sum()

todos_sport_count_norm = list_sport_count_norm[0]
for ps in list_sport_count_norm[1:]:
    todos_sport_count_norm = todos_sport_count_norm.add(ps, fill_value=0)
todos_sport_count_norm = todos_sport_count_norm/todos_sport_count_norm.sum()

todos_dport_count_raw.plot(kind="bar", rot=45); plt.show()
todos_dport_count_norm.plot(kind="bar", rot=45); plt.show()
# pero los count no se pueden binear así ...
# igual, se puede muestrear basado en los count y discretizar el muestreo.?
# pd.Series(np.random.choice(todos_count_norm.index.values, size=1000000, p=todos_count_norm.values)).value_counts(normalize=True).sort_index()

dportsynt=pd.Series(np.random.choice(todos_dport_count_norm.index.values, size=1000000, p=todos_dport_count_norm.values))
dportsynth_bins = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(dportsynt.values.reshape(-1,1)).bin_edges_[0].astype(np.int32)

sportsynt=pd.Series(np.random.choice(todos_sport_count_norm.index.values, size=1000000, p=todos_sport_count_norm.values))
sportsynth_bins = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(sportsynt.values.reshape(-1,1)).bin_edges_[0].astype(np.int32)

print("dport_bins = ", dportsynth_bins)
print("sport_bins = ", sportsynth_bins)



list_dport_all = pd.concat(list_dport)
discr = KBinsDiscretizer(n_bins=10, strategy="quantile").fit(list_dport_all.values.reshape(-1,1))
In [435]: bins=discr.bin_edges_[0]
     ...: 
     ...: bins=np.insert(bins, 0, -1)
     ...: bins=np.append(bins, 2**16)  # o sustutuir el ultimo por 2**16.
# dport array([   -1.,     0.,   443.,  1883., 35544., 40472., 46456., 48752., 54893., 65529., 65536.])
# dport array([   -1.,     0.,   443.,  1883., 35544., 40472., 46456., 48752., 54893., 65536.])
# sport 
pd.cut(list_dport_all.values, bins=np.unique(np.sort(selected_bins))).value_counts().plot(kind="bar", rot=40); plt.show()