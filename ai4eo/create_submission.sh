#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder

#python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --input_channels 1 
python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1  --input_channels 4 --max-epochs 2 --learning-rate 1e-5
tar -C $1 -zcvf submission.tar.gz . 
#tar -zcvf submission.tar.gz . 
