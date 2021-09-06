#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model /home/dkrz/k202143/nni-experiments/dx6gnrOs/trials/jJPoX/best_model.pt --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 --max-epochs 500 --learning-rate 1e-4 --batch-size 15 --patience 50 --n-blocks 60 --n-channels 58
tar -C $1 -zcvf submission.tar.gz . 
#tar -zcvf submission.tar.gz . 
