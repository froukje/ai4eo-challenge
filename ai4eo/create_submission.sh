#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /home/dkrz/k202143/nni-experiments/5DwK6OfC/trials/ho9h1/best_model.pt --bands B02 B03 B04 B08 --max-epochs 500 --learning-rate 1e-4 --patience 50 --n-blocks 54 --n-channels 102 --n-time-frames 25
tar -C $1 -zcvf submission_5DwK6OfC_ho9h1.tar.gz . 
#tar -zcvf submission.tar.gz . 
