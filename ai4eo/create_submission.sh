#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --trained-model best_model_wkn.pt --bands B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 --n-blocks 24 --n-channels 116 --flag train
tar -C $1 -zcvf submission_TV.tar.gz . 
#tar -zcvf submission.tar.gz . 
