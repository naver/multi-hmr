#!/bin/bash

data_ids=("130")
det_thresh_values=(0.05 0.1 0.15 0.2 0.25)
total_gpus=8
i=0

for data_id in "${data_ids[@]}"; do
    for camera in front rear right left; do
        for det_thresh in "${det_thresh_values[@]}"; do
            gpu_id=$((i % total_gpus)) 

            if [ ! -d "output/output_${data_id}_${det_thresh}" ]; then
                mkdir "output/output_${data_id}_${det_thresh}"
            fi

            if [ ! -d "output/output_${data_id}_${det_thresh}/camera_${camera}" ]; then
                mkdir "output/output_${data_id}_${det_thresh}/camera_${camera}"
            fi

            CUDA_VISIBLE_DEVICES=$gpu_id python demo.py --img_folder "${data_id}/camera_${camera}" --out_folder "output/output_${data_id}_${det_thresh}/camera_${camera}" --det_thresh "$det_thresh" &

            ((i++))
            if (( i % (total_gpus * 4) == 0 )); then
                wait
            fi
        done
    done
done

wait

echo "All done"