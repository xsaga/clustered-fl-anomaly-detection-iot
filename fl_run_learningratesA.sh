#!/usr/bin/env bash
set -e

# Run the Federated Learning training.
# Grid search to fine tune the learning rates.

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

FL_ROUNDS=60

client_adam1lr0p0001_args="-d 69 --clientopt adam --lr 0.0001 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p0005_args="-d 69 --clientopt adam --lr 0.0005 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p001_args="-d 69 --clientopt adam --lr 0.001 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p005_args="-d 69 --clientopt adam --lr 0.005 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p01_args="-d 69 --clientopt adam --lr 0.01 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p05_args="-d 69 --clientopt adam --lr 0.05 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam1lr0p1_args="-d 69 --clientopt adam --lr 0.1 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"

server_sgdlr0p5_args="-d 69 -e 1 --serveropt sgd --lr 0.5 --wdecay 0 --momentum 0"
server_sgdlr0p75_args="-d 69 -e 1 --serveropt sgd --lr 0.75 --wdecay 0 --momentum 0"
server_sgdlr1p0_args="-d 69 -e 1 --serveropt sgd --lr 1.0 --wdecay 0 --momentum 0"
server_sgdlr1p25_args="-d 69 -e 1 --serveropt sgd --lr 1.25 --wdecay 0 --momentum 0"
server_sgdlr1p5_args="-d 69 -e 1 --serveropt sgd --lr 1.5 --wdecay 0 --momentum 0"

# server_sgdm_args="-d 69 -e 1 --serveropt sgd --lr 1 --wdecay 0 --momentum 0.9"
# server_adam1_args="-d 69 -e 1 --serveropt adam --lr 1e-2 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 0"
# server_adam2_args="-d 69 -e 1 --serveropt adam --lr 1e-2 --b1 0.9 --b2 0.99 --eps 1e-3 --wdecay 0"


trial_count=1

for client_opt_args in client_adam1lr0p0001_args client_adam1lr0p0005_args client_adam1lr0p001_args client_adam1lr0p005_args client_adam1lr0p01_args client_adam1lr0p05_args client_adam1lr0p1_args
do
    # parse client optimizer name
    client_opt=$(echo ${client_opt_args} | awk -F "_" '{print $2}')

    for server_opt_args in server_sgdlr0p5_args server_sgdlr0p75_args server_sgdlr1p0_args server_sgdlr1p25_args server_sgdlr1p5_args
    do
        # parse server optimizer name
        server_opt=$(echo ${server_opt_args} | awk -F "_" '{print $2}')

        experiment_name=lrgridsearchA${trial_count}_${client_opt}_${server_opt}


        ##### Experiment
        echo "***** Trial $trial_count ***** Directory: $experiment_name"
        # ${!var}  =>  bash indirect expansion
        echo ${client_opt_args}: ${!client_opt_args}
        echo ${server_opt_args}: ${!server_opt_args}

        python fl_server.py -b "$experiment_name" ${!server_opt_args}

        for i in $(seq $FL_ROUNDS); do
            echo "======= FL round $i start ======="
            parallel --verbose --bar --jobs 6 python fl_client.py -b "$experiment_name" ${!client_opt_args} {1} ::: *.pcap
            sleep 1
            python fl_server.py -b "$experiment_name" ${!server_opt_args}
            echo "======= FL round $i end ======="
            sleep 5
        done

        echo "DONE $experiment_name."


        trial_count=$((trial_count+1))
    done
done
