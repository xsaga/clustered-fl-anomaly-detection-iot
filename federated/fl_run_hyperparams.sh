#!/usr/bin/env bash
set -e

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

FL_ROUNDS=100

client_sgd_args="-d 69 --clientopt sgd --lr 1e-3 --wdecay 1e-5 --momentum 0"
client_sgdm_args="-d 69 --clientopt sgd --lr 1e-3 --wdecay 1e-5 --momentum 0.9"
client_adam1_args="-d 69 --clientopt adam --lr 1e-3 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 1e-5"
client_adam2_args="-d 69 --clientopt adam --lr 1e-3 --b1 0.9 --b2 0.99 --eps 1e-3 --wdecay 1e-5"

server_sgd_args="-d 69 -e 1 --serveropt sgd --lr 1 --wdecay 0 --momentum 0"
server_sgdm_args="-d 69 -e 1 --serveropt sgd --lr 1 --wdecay 0 --momentum 0.9"
server_adam1_args="-d 69 -e 1 --serveropt adam --lr 1e-2 --b1 0.9 --b2 0.999 --eps 1e-08 --wdecay 0"
server_adam2_args="-d 69 -e 1 --serveropt adam --lr 1e-2 --b1 0.9 --b2 0.99 --eps 1e-3 --wdecay 0"


trial_count=1

for client_opt_args in client_sgd_args client_sgdm_args client_adam1_args client_adam2_args
do
    # parse client optimizer name
    client_opt=$(echo ${client_opt_args} | awk -F "_" '{print $2}')

    for server_opt_args in server_sgd_args server_sgdm_args server_adam1_args server_adam2_args
    do
        # parse server optimizer name
        server_opt=$(echo ${server_opt_args} | awk -F "_" '{print $2}')

        experiment_name=trial${trial_count}_${client_opt}_${server_opt}


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
