#!/bin/bash

BASE_URL="https://cernbox.cern.ch/remote.php/dav/public-files/HaQVugVTlzbKEXH/unflipped"
OUTDIR="labels"

mkdir -p "$OUTDIR"

for i in $(seq 16401 16441); do
    FILE="labels_d${i}.parquet"
    echo "Downloading $FILE"
    wget -c \
        -O "${OUTDIR}/${FILE}" \
        "${BASE_URL}/${FILE}"
done

for i in $(seq 16401 16441); do
    FILE="recon2D_d${i}.parquet"
    echo "Downloading $FILE"
    wget -c \
        -O "${OUTDIR}/${FILE}" \
        "${BASE_URL}/${FILE}"
done
