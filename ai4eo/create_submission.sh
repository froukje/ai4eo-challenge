#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder

python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --target-dir $1 --input_channels 4 --bands 'B01' 'B02' 'B03' 
tar -C $1 -zcvf submission.tar.gz . 
#tar -zcvf submission.tar.gz . 
