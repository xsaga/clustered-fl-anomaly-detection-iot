from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams


localepochs = 4  # 4
cluster_fname = "cluster_cam"
global_models_dir_path = Path(f"./{cluster_fname}/E_{localepochs}/models_global")
nofl_dir = global_models_dir_path.parent / "./nofl"

global_round_dirs = [p for p in global_models_dir_path.iterdir() if p.is_dir() and p.match("round_*")]
global_round_dirs = sorted(global_round_dirs, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

# FL
fl_weighted_losses = []
fl_losses_min = []
fl_losses_max = []
fl_weighted_train_losses = []
fl_losses_raw = []
for d in global_round_dirs[1:]:
    loss_f = [f for f in d.iterdir() if f.is_file() and f.match("losses_num_samples_round*.npz")][0]
    fl_losses, fl_train_losses, num_samples = np.load(loss_f).values()
    fl_losses_raw.append(fl_losses)
    fl_weighted_losses.append(np.average(fl_losses, weights=num_samples))
    fl_losses_min.append(np.min(fl_losses))
    fl_losses_max.append(np.max(fl_losses))
    fl_weighted_train_losses.append(np.average(fl_train_losses, weights=num_samples))

plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_train_losses, color="red")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_train_losses, color="red")
plt.yscale("log")
plt.show()

fl_losses_raw = np.array(fl_losses_raw)  # num_rounds x num_clients
fl_losses_raw = fl_losses_raw.transpose((1, 0))  # num_clients x num_rounds

# no FL
nofl_files = [p for p in nofl_dir.iterdir() if p.is_file() and p.match("results_nofl_epoch_losses_*")]
losses = []
for f in nofl_files:
    epochs, loss, val_loss = np.load(f).values()
    losses.append(val_loss)
losses = np.array(losses)  # num_clients x num_epochs
min_loss = np.min(losses, axis=0)
max_loss = np.max(losses, axis=0)
avg_loss = np.mean(losses, axis=0)


# plt.scatter(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue")
plt.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue", label="Avg FL loss")
# plt.fill_between(list(range(1, len(global_round_dirs))), fl_losses_max, fl_losses_min, alpha=0.3, color="blue", hatch="\\", label="Min to Max FL loss range")
plt.boxplot(fl_losses_raw, showmeans=True, patch_artist=True, boxprops=dict(facecolor="blue", edgecolor="blue", hatch="\\", alpha=0.2), capprops=dict(color="blue"), whiskerprops=dict(color="blue"), flierprops=dict(color="blue", markeredgecolor="blue", alpha=0.2), medianprops=dict(color="blue"), meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"))
# plt.scatter(list(range(1, len(epochs[4::5])+1)), avg_loss[4::5], color="green")
plt.plot(list(range(1, len(epochs[(localepochs - 1)::localepochs]) + 1)), avg_loss[(localepochs - 1)::localepochs], color="green", label=f"Avg isolated loss every {localepochs} rounds")
# plt.fill_between(list(range(1, len(epochs[4::5])+1)), max_loss[4::5], min_loss[4::5], alpha=0.3, color="green", hatch="/", label="Min to Max isolated loss range")
plt.boxplot(losses[:, (localepochs - 1)::localepochs], showmeans=True, patch_artist=True, boxprops=dict(facecolor="green", edgecolor="green", hatch="/", alpha=0.2), capprops=dict(color="green"), whiskerprops=dict(color="green"), flierprops=dict(color="green", markeredgecolor="green", alpha=0.2), medianprops=dict(color="green"), meanprops=dict(markerfacecolor="green", markeredgecolor="green"))
plt.yscale("log")
plt.xlabel("Training rounds")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.set_xlim((1, len(global_round_dirs) - 1))
ax.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color="blue", label="Avg FL loss")

ax.boxplot(fl_losses_raw, showmeans=True, patch_artist=True, boxprops=dict(facecolor="blue", edgecolor="blue", hatch="\\", alpha=0.2), capprops=dict(color="blue"), whiskerprops=dict(color="blue"), flierprops=dict(color="blue", markeredgecolor="blue", alpha=0.2), medianprops=dict(color="blue"), meanprops=dict(markerfacecolor="blue", markeredgecolor="blue"))

ax2 = ax.twiny()
ax2.set_xlim((localepochs, len(epochs)))
ax2.plot(epochs + 1, avg_loss[::], color="green", label="Avg isolated loss")

ax2.boxplot(losses[:, ::], showmeans=True, patch_artist=True, boxprops=dict(facecolor="green", edgecolor="green", hatch="/", alpha=0.2), capprops=dict(color="green"), whiskerprops=dict(color="green"), flierprops=dict(color="green", markeredgecolor="green", alpha=0.2), medianprops=dict(color="green"), meanprops=dict(markerfacecolor="green", markeredgecolor="green"))
plt.yscale("log")

ax2labels = [str(lbl) if lbl % localepochs == 0 else "" for lbl in epochs + 1]
ax2.set_xticklabels(ax2labels)

ax.grid(axis="x")
ax.set_xlabel("Training rounds")
ax.set_ylabel("Loss")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.tight_layout()
plt.show()


rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.linewidth"] = 0.75
rcParams["lines.markersize"] = 4

# rcParams["lines.linewidth"] = 0.75
# rcParams["lines.markersize"] = 1
# rcParams["boxplot.boxprops.linewidth"] = 0.1 # 1
# rcParams['boxplot.whiskerprops.linewidth'] = 0.1 # 1
# rcParams["boxplot.capprops.linewidth"] = 0.1 # 1
# rcParams["boxplot.meanprops.linewidth"] = 0.1 # 1
# rcParams["boxplot.medianprops.linewidth"] = 0.1 # 1
# rcParams["boxplot.flierprops.markeredgewidth"] = 1 # 1
# rcParams["boxplot.flierprops.markersize"] = 1 # 6
# rcParams["boxplot.meanprops.markersize"] = 1 # 6
# rcParams["xtick.major.size"] = 3.5//2 # 3.5
# rcParams["ytick.major.size"] = 3.5//2 # 3.5
# rcParams["xtick.minor.size"] = 2//2 # 2.0
# rcParams["ytick.minor.size"] = 2//2 # 2.0
plot_width = 3.487  # in
plot_height = 3.487


# === jumps in nofl boxplot === (simplify plot)
epoch_mask = np.array([x % localepochs == 0 for x in range(1, losses.shape[-1] + 1)])
losses[:, ~epoch_mask] = np.nan

fig, ax = plt.subplots()
ax.set_xlim((1, len(global_round_dirs) - 1))

ax2 = ax.twiny()
ax2.set_xlim((localepochs, len(epochs)))

nofl_color = "#ff7f0e"
nofl_line, = ax2.plot(epochs + 1, avg_loss[::], color=nofl_color, marker="", label="Isolated edge mean loss")
nofl_bp = ax2.boxplot(losses[:, ::], widths=(0.5 * localepochs), showmeans=True, patch_artist=True, boxprops=dict(facecolor=nofl_color, edgecolor=nofl_color, hatch="/", alpha=0.4), capprops=dict(color=nofl_color), whiskerprops=dict(color=nofl_color), flierprops=dict(color=nofl_color, marker="v", markeredgecolor=nofl_color, alpha=0.3), medianprops=dict(color=nofl_color), meanprops=dict(markerfacecolor=nofl_color, marker="v", markeredgecolor=nofl_color))
#        "           (losses[:,::], widths=2.0, showmeans=True,

fl_color = "#1f77b4"
fl_line, = ax.plot(list(range(1, len(global_round_dirs))), fl_weighted_losses, color=fl_color, marker="^", label="FL weighted mean loss")
fl_bp = ax.boxplot(fl_losses_raw, showmeans=False, patch_artist=True, boxprops=dict(facecolor=fl_color, edgecolor=fl_color, hatch="\\", alpha=0.4), capprops=dict(color=fl_color), whiskerprops=dict(color=fl_color), flierprops=dict(color=fl_color, marker="^", markeredgecolor=fl_color, alpha=0.3), medianprops=dict(color=fl_color), meanprops=dict(markerfacecolor=fl_color, marker="^", markeredgecolor=fl_color))


plt.yscale("log")

#                     l % 2 == 0
axlabels = [str(lbl) if lbl % 5 == 0 else "" for lbl in range(1, len(global_round_dirs))]
axlabels[0] = str(range(1, len(global_round_dirs))[0])  # == "1"
ax.set_xticklabels(axlabels)

ax2labels = [str(lbl) if lbl % (localepochs * 10) == 0 else "" for lbl in epochs + 1]  # l%(localepochs*2)==0
first_ax2label = epoch_mask.nonzero()[0][0]
ax2labels[first_ax2label] = str(first_ax2label + 1)
ax2.set_xticklabels(ax2labels)

# ax.grid(axis="x")
ax.set_xlabel("FL rounds")
ax.set_ylabel("evaluation loss")
ax2.set_xlabel("isolated edge training epochs")

nofl_line_legend = plt.Line2D([0], [0])
nofl_line_legend.update_from(nofl_line)
nofl_line_legend.set_marker("v")
fig.legend([fl_bp["boxes"][0], fl_line, nofl_bp["boxes"][0], nofl_line_legend], ["FL", fl_line.get_label(), "Isolated edge", nofl_line_legend.get_label()], loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
fig.set_size_inches(plot_width, plot_height)
plt.tight_layout()
ax.set_zorder(1)
ax.set_frame_on(False)
# plt.show()
plt.savefig(f"{cluster_fname}_e{localepochs}_boxplot.pdf", format="pdf")

####
# for i in range(len(nofl_files)):
#     print(nofl_files[i])
#     e, l, vl = np.load(nofl_files[i]).values()
#     plt.plot(e, l, color="blue", label="train loss")
#     plt.plot(e, vl, color="red", label="validation loss")
#     plt.title(nofl_files[i])
#     plt.legend(); plt.show()
