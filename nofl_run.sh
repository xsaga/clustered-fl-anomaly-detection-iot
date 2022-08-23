#!/usr/bin/env bash
set -e

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

ROUNDS=100
opt_args="-d 69 --opt adam --lr 0.001 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"

for TIMES in 1 2 4 8
do

    experiment_name=E_${TIMES}/nofl
    epochs=$((ROUNDS*TIMES))

    ##### Experiment
    echo "***** epochs = $epochs ***** Directory: $experiment_name"
    echo "Client: ${client_opt_args}"

    parallel --verbose --bar --jobs 6 python nofl_client_loss.py -b "$experiment_name" -e $epochs ${opt_args} {1} ::: *.pcap

    echo "DONE $experiment_name."
    sleep 5

done
