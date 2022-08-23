#!/usr/bin/env bash
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

for EPSILON in 1 2 4 8 16 32
do
    echo "========== $EPSILON =========="
    mkdir clus_${EPSILON}epochs_port_hier_iot
    parallel --verbose --bar --ungroup --jobs 10 python {1} -d 69 -e $EPSILON {2} ::: cluster_client_ae.py ::: *.pcap
    sleep 5
    mv -- *_ae.pt clus_${EPSILON}epochs_port_hier_iot
    cp clus_${EPSILON}epochs_port_hier_iot/initial_random_model_ae.pt .
done

date '+%s' > end.timestamp
