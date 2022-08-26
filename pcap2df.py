"""Apply the 'pcap_to_dataframe' function to pcap files and save the resulting DataFrame.

Transforming the pcap to a DataFrame can take a long time for large
pcap files. Use this script to precompute the DataFrames and use them
in the other programs.
"""
import argparse
from pathlib import Path

from feature_extractor import pcap_to_dataframe


parser = argparse.ArgumentParser(description="pcap file to pandas DataFrame")
parser.add_argument("pcap", type=lambda p: Path(p).absolute(), nargs="+")
args = parser.parse_args()

for pcap_file in args.pcap:
    print(f"Processing pcap from: {pcap_file}")
    df = pcap_to_dataframe(pcap_file.as_posix(), verbose=True)

    out_file = pcap_file.with_suffix(".pickle")
    print(f"Saving DataFrame to: {out_file}\n")
    df.to_pickle(out_file)
