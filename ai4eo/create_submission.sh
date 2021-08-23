#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder

python submission.py  --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --filters 32 --target-dir $1
tar -C $1 -zcvf submission.tar.gz . 
