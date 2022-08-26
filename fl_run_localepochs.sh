#!/usr/bin/env bash
set -e

# Run the Federated Learning training.
# Multiple experiments for different number of local training epochs.

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

FL_ROUNDS=100

client_opt_args="-d 69 --clientopt adam --lr 1e-3 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
server_opt_args="-d 69 --serveropt sgd --lr 1 --wdecay 0 --momentum 0"

for epochs in 1 2 4 8
do

    experiment_name=E_${epochs}

    ##### Experiment
    echo "***** E = $epochs ***** Directory: $experiment_name"

    echo "Client: ${client_opt_args}"
    echo "Server: ${server_opt_args}"

    python fl_server.py -b "$experiment_name" -e $epochs ${server_opt_args}

    for i in $(seq $FL_ROUNDS); do
        echo "======= FL round $i start ======="
        parallel --verbose --bar --jobs 6 python fl_client.py -b "$experiment_name" ${client_opt_args} {1} ::: *.pcap
        sleep 1
        python fl_server.py -b "$experiment_name" -e $epochs ${server_opt_args}
        echo "======= FL round $i end ======="
        sleep 5
    done

    echo "DONE $experiment_name."

done
