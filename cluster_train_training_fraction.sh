#!/usr/bin/env bash

# Additional experiment to verify the clustering performance when the
# fraction of training data of each device is changed.

# Run the partial training of the models for the device clustering
# step. If needed, adjust the input dimensions of the autoencoder model
# (change cluster_client_ae.py accordingly). Models are trained in
# parallel, adjust the number of parallel jobs according to the number
# of cores.

set -e

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

if [ -f "initial_random_model_ae.pt" ]; then
    echo "initial_random_model_ae.pt exists."
else
    echo "Creating initial_random_model_ae.pt"
    python create_initial_random_model_ae.py 69
fi

echo "$(ls -- *.pcap | wc -l)" training files found.

date '+%s' > start.timestamp

# Fix epsilon (number of training epochs)
EPSILON=4

for FRACTION in 0.01 0.1 0.2 0.4 0.6 0.8 0.99
do
    echo "========== $FRACTION =========="
    mkdir clus_${EPSILON}epochs_port_hier_iot_${FRACTION}frac
    parallel --verbose --bar --ungroup --jobs 10 python {1} -d 69 -e $EPSILON -f $FRACTION {2} ::: cluster_client_ae.py ::: *.pcap
    sleep 5
    mv -- *_ae.pt clus_${EPSILON}epochs_port_hier_iot_${FRACTION}frac
    mv clus_${EPSILON}epochs_port_hier_iot_${FRACTION}frac/initial_random_model_ae.pt .
done

date '+%s' > end.timestamp
