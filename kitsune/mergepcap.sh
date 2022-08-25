#!/usr/bin/env bash

# Concatenates two pcap files. The timestamps of the packets in the
# first pcap file are shifted so that the timestamp of the last packet
# from the first pcap is just before the first packet of the second pcap
# file (no time gap between the two) so that it appears as a single
# capture.

# Used to evaluate pcaps with Kitsune, because it is sensitive to packet
# timings.

set -e

FIRSTPCAP=$1
LASTPCAP=$2

echo "Info:"
capinfos -c "$FIRSTPCAP"
capinfos -c "$LASTPCAP"

echo "Merging: $FIRSTPCAP <---> $LASTPCAP"

# last pcap start time
STARTTIME=$(capinfos -a "$LASTPCAP" | grep '^First packet time' | cut -d ':' -f 2- | xargs echo -n)
# first pcap end time
ENDTIME=$(capinfos -e "$FIRSTPCAP" | grep '^Last packet time' | cut -d ':' -f 2- | xargs echo -n)

echo "Start time: $STARTTIME"
echo "End time  : $ENDTIME"

DELTASECS=$(($(date -d "$STARTTIME" +%s) - $(date -d "$ENDTIME" +%s)))
echo "Delta time = $DELTASECS seconds"
if [ "$DELTASECS" -lt 0 ]; then
    echo "Negative delta time. Exiting."
    exit 1
fi

echo "Shifting $FIRSTPCAP by $DELTASECS seconds"
editcap -t "$DELTASECS" "$FIRSTPCAP" shiftedfirst.pcap

echo "Merging pcaps"
mergecap -a shiftedfirst.pcap "$LASTPCAP" -w merged.pcap

rm shiftedfirst.pcap
