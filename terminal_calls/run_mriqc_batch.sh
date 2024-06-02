#!/bin/bash

# Check if a delay argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <delay_in_minutes>"
    exit 1
fi

# Convert delay to seconds for sleep command
delay=$(($1 * 60))

echo "Waiting for $1 minute(s) before starting..."
sleep $delay

# Iterate over subject numbers
for i in {0..40}; do
    # Skip subject 0, 5, 14, 31
    if [ $i -eq 0 ] || [ $i -eq 5 ] || [ $i -eq 14 ] || [ $i -eq 31 ]; then
        continue
    fi

    # Format subject number with leading zeros
    subID=$(printf "sub-%02d" $i)
    echo "Processing $subID"

    # Docker command with dynamic subject ID using sudo
    docker run -it --rm \
    -v /data/projects/chess/data/BIDS:/data:ro \
    -v /data/projects/chess/data/BIDS/derivatives/mriqc:/out \
    -v /home/eik-tb/Desktop/temp_mriqc:/scratch \
    nipreps/mriqc:latest /data /out participant \
    --participant-label ${subID} \
    --nprocs 16 --mem-gb 40 --float32 \
     --work-dir /scratch \
     --verbose-reports --resource-monitor -vv

    # Wait or perform other actions between runs if needed
    sleep 0.5
done

echo "Running group analysis"

docker run -it --rm \
-v /data/projects/chess/data/BIDS:/data:ro \
-v /data/projects/chess/data/BIDS/derivatives/mriqc:/out \
-v /home/eik-tb/Desktop/temp_mriqc:/scratch \
nipreps/mriqc:latest /data /out group \
--nprocs 16 --mem-gb 40 --float32 \
 --work-dir /scratch \
 --verbose-reports --resource-monitor -vv

sleep 0.5

echo "Running classifier"
docker run \
-v /home/eik-tb/Desktop/temp_mriqc:/scratch \
-v /data/projects/chess/data/BIDS/derivatives/mriqc:/resdir \
-w /scratch --entrypoint=mriqc_clf poldracklab/mriqc:latest \
 --load-classifier -X /resdir/group_T1w.tsv




