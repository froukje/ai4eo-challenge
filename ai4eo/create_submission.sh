#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /home/dkrz/k202143/nni-experiments/hpebkTd4/trials/EMxGr/best_model.pt --bands B02 B03 B04 B08 --max-epochs 500 --learning-rate 1e-4 --batch-size 12 --patience 50 --n-blocks 8 --n-channels 96
tar -C $1 -zcvf submission_hpebkTd4_EMxGr.tar.gz . 
#tar -zcvf submission.tar.gz . 
