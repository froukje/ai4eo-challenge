#!/usr/bin/bash

# takes the submission directory as an argument
# creates the submission TIFFs
# zips the folder

python submission.py --raw-data-dir /swork/shared_data/2021-ai4eo/eopatches/ --input_channels 1 
#python submission.py  python3 model.py --processed-data-dir /swork/shared_data/2021-ai4eo/dev_data/default/ --input_channels 1 --max-epochs 50
#tar -C $1 -zcvf submission.tar.gz . 
tar -zcvf submission.tar.gz . 
