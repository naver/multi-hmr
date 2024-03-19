#!/bin/bash

data_ids=("052" "055" "056" "059" "060" "061" "062")
total_gpus=8
i=0

for data_id in "${data_ids[@]}"; do
    for camera in front rear right left; do    
        gpu_id=$((i % total_gpus)) 

        if [ ! -d "output/output_${data_id}" ]; then
            mkdir "output/output_${data_id}"
        fi

        if [ ! -d "output/output_${data_id}/camera_${camera}" ]; then
            mkdir "output/output_${data_id}/camera_${camera}"
        fi

        CUDA_VISIBLE_DEVICES=$gpu_id python demo.py --img_folder "/mnt/mnt_0/data/inhouse/ovon/training/topics/${data_id}/camera_${camera}" --out_folder "output/output_${data_id}/camera_${camera}" &

        ((i++))
        if (( i % (total_gpus * 4) == 0 )); then
            wait
        fi
    done
done

wait

echo "All done"