#!/usr/bin/env bash
find ./clustering_compromised -type d -iname "clus_4epochs_re*" -exec python cluster_compromised_evaluation.py --dir {} --dimensions 69 \;
