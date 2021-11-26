import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

fl_paths = ["E_2/models_global",
            "E_4/models_global",
            #"E_4_seed1/models_global",
            "E_8/models_global"]

def get_losses_raw(dir_path):
    global_models_dir_path = Path(dir_path)

    global_round_dirs = [p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
    global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    fl_losses_raw = []
    for d in global_round_dirs[1:]:
        loss_f = [f for f in d.iterdir() if f.is_file() and f.match("losses_num_samples_round*.npz")][0]
        fl_losses, fl_train_losses, num_samples = np.load(loss_f).values()
        fl_losses_raw.append(fl_losses)

    fl_losses_raw = np.array(fl_losses_raw)
    fl_losses_raw=fl_losses_raw.transpose((1,0))
    return fl_losses_raw

losses_dic = {}
for p in fl_paths:
    losses_dic[p.split("/")[0]] = get_losses_raw(p)

sns.boxplot(data=losses_dic["E_2"], color="blue")
sns.boxplot(data=losses_dic["E_4"], color="green")
#sns.boxplot(data=losses_dic["E_4_seed1"], color="green")
sns.boxplot(data=losses_dic["E_8"], color="red")
plt.legend(["E2", "E4", "E8"])
plt.yscale("log")
plt.show()