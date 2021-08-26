## How to get started with singularity on vader

Documentation for using singularity on mistral:
https://www.dkrz.de/up/systems/mistral/singularity

### Create the docker image

As explained in the documentation, it is not possible to directly create the image directly on mistral as it requires sudo. I created the docker image on my laptop (ubuntu 20.04 running via wsl 2)


1. Start the docker daemon: `sudo dockerd`
2. Checkout the git repository: `git checkout git@gitlab.dkrz.de:k202141/ai4eo-challenge.git`
3. `cd ai4eo-challenge`
4. Create the docker image: `sudo docker build -f Dockerfile-jupyter-user-g-stripped-cuda11 -t ai4eo .`

### Transfer the image to mistral

You now have a local docker image. To transfer it to mistral, I used dockerhub: `hub.docker.com`. Register a user account (`$USER_DOCKERHUB`) and do the following:

1. `sudo docker tag ai4eo2 $USER_DOCKERHUB/ai4eo2` 
2. `sudo docker push $USER_DOCKERHUB/ai4eo`

Login to mistral. Checkout the git repository there as well:

1. Checkout the git repository: `git@gitlab.dkrz.de:k202141/ai4eo-challenge.git`
2. `cd ai4eo-challenge`

Create the singularity image:

1. Start an interactive session on any node with internet access (GPUs on mistral / trial)
2. Activate singularity module: `module load singularity`
3. Pull the image `singularity pull docker://$USER_DOCKERHUB/ai4eo`

## Run the singularity container

Create an allocation for an interactive job on any of the amd nodes

1. `ssh trial.dkrz.de`
2. `salloc`
3. `ssh vader{N}`
4. Start the singularity container, taking care of the following: `--nv` for nvidia GPUs, bind cache to a writable folder (I used scratch space), bind shared data so it is visible when you work in the container: `singularity shell --nv --bind /scratch/k/$USER/singularity/cache:/home/jovyan/.cache --bind /mnt/lustre02/work/ka1176/:/swork /work/ka1176/caroline/gitlab/ai4eo-challenge/ai4eo_latest.sif`

The `/work/ka1176/` folder is now available at `/swork/`

Now we need to activate the conda environment:

TODO

1. opt ...
2. list the environments
3. conda activate the environment


## Run the model 

TODO

## Create the submission file

TODO
