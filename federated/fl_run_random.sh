#!/usr/bin/env bash
set -e

# from sort coreutils info pages
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
}
     

# sort --random-sort --random-source=<(get_seeded_random 42)
# ls *.pcap | sort --random-sort --random-source=<(get_seeded_random 42) | head -n 5

command -v parallel >/dev/null 2>&1 || { echo >&2 "Install GNU Parallel.  Aborting."; exit 1; }
python -c "import torch; print(torch.__version__)" || { echo >&2 "Configure your python environment. Aborting."; exit 1; }

# clean dirs?
if [[ -n $(find models_local -iname "*.tar" -type f) ]]; then
    echo "not empty. Aborting."
    exit 1
else
    echo "is empty"
fi

FL_ROUNDS=4

python fl_server.py

for i in $(seq $FL_ROUNDS); do
    echo ======= FL round $i start =======
    # parallel --verbose --bar --jobs 8 python fl_client.py {1} ::: *.pcap
    parallel --verbose --bar --jobs 8 python fl_client.py {1} :::: <(ls *.pcap | sort --random-sort --random-source=<(get_seeded_random 42) | head -n 50)
    sleep 1
    python fl_server.py
    echo ======= FL round $i end =======
    sleep 5
done

echo DONE.
