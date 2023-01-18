#TODO rename
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def populate_directory(target_dir: Path, normal_dir: Path, compromised_dir: Path, num_compromised_per_cluster: int, devices_per_cluster: List[List[str]]) -> Path:
    # Use symlinks?
    target_dir.mkdir(exist_ok=True)

    compromised_files = []

    # populate normal files in target directory
    for cluster in devices_per_cluster:
        for device in cluster:
            device_files = list(normal_dir.glob(f"{device}_*.pickle"))
            assert len(device_files) == 1
            shutil.copy2(device_files[0], target_dir)

    # replace normal files with compromised devices
    if num_compromised_per_cluster > 0:
        for cluster in devices_per_cluster:
            for device in cluster[:num_compromised_per_cluster]:
                device_files = list(compromised_dir.glob(f"{device}_*.pickle"))
                assert len(device_files) == 1
                shutil.copy2(device_files[0], target_dir)
                compromised_files.append(device)

    # write metadata
    with open(target_dir / "README.txt", "w", encoding="utf-8") as metadatafile:
        metadatafile.write(f"Number of compromised devices included: {len(compromised_files)}\n")
        if compromised_files:
            metadatafile.writelines([f"* {c}\n" for c in compromised_files])

    assert len(list(target_dir.glob("*.pickle"))) == sum(len(d) for d in devices_per_cluster)
    return target_dir


devices_in_cluster = [
# "Cluster 0"

["iotsim-air-quality-1",
"iotsim-building-monitor-1",
"iotsim-building-monitor-2",
"iotsim-building-monitor-3",
"iotsim-building-monitor-4",
"iotsim-building-monitor-5",
"iotsim-domotic-monitor-1",
"iotsim-domotic-monitor-2",
"iotsim-domotic-monitor-3",
"iotsim-domotic-monitor-4",
"iotsim-domotic-monitor-5"],

# "Cluster 1"

["iotsim-hydraulic-system-1",
"iotsim-hydraulic-system-2",
"iotsim-hydraulic-system-3",
"iotsim-hydraulic-system-4",
"iotsim-hydraulic-system-5",
"iotsim-hydraulic-system-6",
"iotsim-hydraulic-system-7",
"iotsim-hydraulic-system-8",
"iotsim-hydraulic-system-9",
"iotsim-hydraulic-system-10",
"iotsim-hydraulic-system-11",
"iotsim-hydraulic-system-12",
"iotsim-hydraulic-system-13",
"iotsim-hydraulic-system-14",
"iotsim-hydraulic-system-15"],

# "Cluster 2"

["iotsim-city-power-1",
"iotsim-combined-cycle-1",
"iotsim-combined-cycle-2",
"iotsim-combined-cycle-3",
"iotsim-combined-cycle-4",
"iotsim-combined-cycle-5",
"iotsim-combined-cycle-6",
"iotsim-combined-cycle-7",
"iotsim-combined-cycle-8",
"iotsim-combined-cycle-9",
"iotsim-combined-cycle-10"],

# "Cluster 3"

["iotsim-cooler-motor-1",
"iotsim-cooler-motor-2",
"iotsim-cooler-motor-3",
"iotsim-cooler-motor-4",
"iotsim-cooler-motor-5",
"iotsim-cooler-motor-6",
"iotsim-cooler-motor-7",
"iotsim-cooler-motor-8",
"iotsim-cooler-motor-9",
"iotsim-cooler-motor-10",
"iotsim-cooler-motor-11",
"iotsim-cooler-motor-12",
"iotsim-cooler-motor-13",
"iotsim-cooler-motor-14",
"iotsim-cooler-motor-15"],

# "Cluster 4"

["iotsim-ip-camera-museum-1",
"iotsim-ip-camera-museum-2",
"iotsim-ip-camera-street-1",
"iotsim-ip-camera-street-2",
"iotsim-stream-consumer-1",
"iotsim-stream-consumer-2"],

# "Cluster 5"

["iotsim-predictive-maintenance-1",
"iotsim-predictive-maintenance-2",
"iotsim-predictive-maintenance-3",
"iotsim-predictive-maintenance-4",
"iotsim-predictive-maintenance-5",
"iotsim-predictive-maintenance-6",
"iotsim-predictive-maintenance-7",
"iotsim-predictive-maintenance-8",
"iotsim-predictive-maintenance-9",
"iotsim-predictive-maintenance-10"],

# "Cluster 6"

["iotsim-predictive-maintenance-11",
"iotsim-predictive-maintenance-12",
"iotsim-predictive-maintenance-13",
"iotsim-predictive-maintenance-14",
"iotsim-predictive-maintenance-15"],

# "Cluster 7"

["iotsim-combined-cycle-tls-1",
"iotsim-combined-cycle-tls-2",
"iotsim-combined-cycle-tls-3",
"iotsim-combined-cycle-tls-4",
"iotsim-combined-cycle-tls-5"]
]

NORMAL_DATA_DIR = Path("./normal_1sept")
COMPROMISED_DATA_DIR = Path("./merlin_cnc_1sept")
OUTPUT_DIR = Path("./clustering_compromised")
NUMBER_OF_CLUSTERING_INITIALIZATIONS = 10

program_files = [
"initial_random_model_ae.pt",
"create_initial_random_model_ae.py",
"model_ae.py",
"feature_extractor.py",
"cluster_client_ae.py",
"cluster_train.sh"]

for program_file in program_files:
    assert Path(program_file).is_file()

try:
    OUTPUT_DIR.mkdir(exist_ok=False)
except FileExistsError:
    print(f"{OUTPUT_DIR} exists. Exiting.")
    sys.exit(1)

num_devices_in_largest_cluster = max(len(x) for x in devices_in_cluster)

for num_compromised in range(0, num_devices_in_largest_cluster + 1):
    print(f"***{num_compromised} compromised devices for each cluster. Creating directory***")
    out_dir = populate_directory(OUTPUT_DIR / f"trial_{num_compromised}_compromised_per_cluster", NORMAL_DATA_DIR, COMPROMISED_DATA_DIR, num_compromised, devices_in_cluster)

    for program_file in program_files:
        shutil.copy2(program_file, out_dir)

    print("**Running cluster training**")
    for i in range(NUMBER_OF_CLUSTERING_INITIALIZATIONS):
        print(f"*Iter {i+1}*")
        subprocess.run(["bash", "cluster_train.sh", str(i+1)], cwd=out_dir, check=True)
