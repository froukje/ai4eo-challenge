#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /home/dkrz/k202143/nni-experiments/cT3VdxQN/trials/IGshX/best_model.pt --bands B02 B03 B04 B08 --max-epochs 500 --learning-rate 1e-4 --patience 50 --n-blocks 52 --n-channels 117 --n-time-frames 6
tar -C $1 -zcvf submission_cT3VdxQN_IGshX.tar.gz . 
#tar -zcvf submission.tar.gz . 
