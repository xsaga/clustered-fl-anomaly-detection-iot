#!/usr/bin/env bash

# Additional experiment to verify the clustering performance when the
# fraction of training data of each device is changed.

# After completion, run:
# find . -type d -iname "clus_4epochs_port_hier_iot_rep*" -exec python cluster_client_evaluation.py --dir {} --dimensions 69 --clusters 8 --image-format png \;

# EPSILON=4
# mkdir results_all
# for ITERNUM in $(seq 10)
# do
#     for FRACTION in 0.01 0.1 0.2 0.4 0.6 0.8 0.99
#     do
# 	echo "----- REPETITION: $ITERNUM FRACTION: $FRACTION -----"
# 	cp clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac/score_unsupervised.png results_all/score_unsupervised_${FRACTION}frac_rep${ITERNUM}.png
# 	cp clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac/score_groundtruth.png results_all/score_groundtruth_${FRACTION}frac_rep${ITERNUM}.png
# 	cp clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac/cluster_composition.txt results_all/cluster_composition_${FRACTION}frac_rep${ITERNUM}.txt
# 	cp clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac/2d_map.png results_all/2d_map_${FRACTION}frac_rep${ITERNUM}.png
#     done
# done



# Run the partial training of the models for the device clustering
# step. If needed, adjust the input dimensions of the autoencoder model
# (change cluster_client_ae.py accordingly). Models are trained in
# parallel, adjust the number of parallel jobs according to the number
# of cores.

set -e

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

echo "$(ls -- *.pcap | wc -l)" training files found.

date '+%s' > start.timestamp

# Fix epsilon (number of training epochs)
EPSILON=4

for ITERNUM in $(seq 10)
do
    # create random model, inside an if because in the first
    # iteration we might want to use a specific initial model
    # already in the directory.
    if [ -f "initial_random_model_ae.pt" ]; then
	echo "initial_random_model_ae.pt exists."
    else
	echo "Creating initial_random_model_ae.pt"
	python create_initial_random_model_ae.py 69
    fi

    # do clustering experiment
    for FRACTION in 0.01 0.1 0.2 0.4 0.6 0.8 0.99
    do
	echo "========== REPETITION: $ITERNUM FRACTION: $FRACTION =========="
	mkdir clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac
	parallel --verbose --bar --ungroup --jobs 10 python {1} -d 69 -e $EPSILON -f $FRACTION {2} ::: cluster_client_ae.py ::: *.pcap
	sleep 5
	mv -- *_ae.pt clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac
	mv clus_${EPSILON}epochs_port_hier_iot_rep${ITERNUM}_${FRACTION}frac/initial_random_model_ae.pt .
    done
    sleep 5

    # backup random model of this repetition and delete it
    mkdir -p clus_${EPSILON}_initial_random_models/rep_${ITERNUM}
    mv initial_random_model_ae.pt clus_${EPSILON}_initial_random_models/rep_${ITERNUM}
done

date '+%s' > end.timestamp
