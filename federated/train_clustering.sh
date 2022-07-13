#!/usr/bin/env bash
set -e

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

if [ -f "initial_random_model_ae.pt" ]; then
    echo "initial_random_model_ae.pt exists."
else
    echo "Creating initial_random_model_ae.pt"
    python create_initial_random_model_ae.py 27
fi

echo $(ls *.pcap | wc -l) training files found.

date '+%s' > start.timestamp

parallel --verbose --bar --ungroup --jobs 32 python {1} {2} {3} ::: client_cluster_ae.py client_cluster_lstm.py ::: *.pcap ::: 1 2 4 8 16 32

date '+%s' > end.timestamp