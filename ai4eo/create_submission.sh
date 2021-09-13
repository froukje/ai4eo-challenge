#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /swork/caroline/jobs/ai4eo/all_indices_random/epoch_120_model.pt --bands B02 B03 B04 B08 --n-blocks 52 --n-channels 117 --n-time-frames 6 
tar -C $1 -zcvf $1.tar.gz . 
#tar -zcvf submission.tar.gz . 
