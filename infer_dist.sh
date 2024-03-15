#!/bin/bash

for data_id in $(seq -w 062 062)
do
    if [ ! -d output_"${data_id}" ]; then
        mkdir output_"${data_id}"
    fi

    for camera in front rear right left
    do    
        if [ ! -d output_"${data_id}"/camera_"${camera}" ]; then
            mkdir output_"${data_id}"/camera_"${camera}"
        fi
        python demo.py --img_folder "/mnt/mnt_0/data/inhouse/ovon/training/topics/${data_id}/camera_${camera}" --out_folder "output_${data_id}/camera_${camera}" &
    done
    wait
    echo "Processing batch ${data_id} done"
done

echo "All done"
