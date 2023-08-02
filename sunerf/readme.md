# Environment setup

Create a conda environment (or use pyenv, etc)

```bash
conda create -n helio-ground python=3.10
conda activate helio-ground
```

and install the dependencies:

```bash
pip install -e .[ground]
```

This should install torch with CUDA support, but you should double check that the version installed is compatible with your base system.

# Getting pre-prepared data

Minimally, pull the pre-prepped data from the bucket:

```bash
gsutil -m rsync -r gs://fdl23_europe_helio_onground/prep-data ./data/prep-data
```

This contains a number of folders with various simulation setups using different viewpoint configurations.

# Train

To train a model, run:

```bash
python -m sunerf.sunerf --data_path_pB "data/prep-data/prep_HAO_1view/*pB*.fits" --data_path_tB "data/prep-data/prep_HAO_1view/*tB*.fits" --path_to_save "results/training/HAO_1view" --train "config/train.yaml" --hyperparameters "config/hyperparams_icarus.yaml"
```

You can modify the batch size in `train.yaml` if you need, for example on an RTX3090, a batch size of 2048 will fit into memory and will take around XXX minutes per epoch for this configuration (0.5s / batch)

# Run in Docker:

First install the [Nvidia container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Then build the container using the script - note that we pull the repo into the container, so don't just run docker build directly:

```bash
cd docker/ground
./build.sh
```

Then mount the data and output folders and run, e.g. from the root directory:

```bash
docker run --gpus all -it --rm -v data:/icarus/data -v output:/icarus/output -v results:/icarus/results helio-ground --data_path_pB "data/prep-data/prep_HAO_1view/*pB*.fits" --data_path_tB "data/prep-data/prep_HAO_1view/*tB*.fits" --path_to_save "results/training/HAO_1view" --train "config/train.yaml" --hyperparameters "config/hyperparams_icarus.yaml"
```

You'll be asked to login to wandb or you can ignore this and train without it.

# Evaluate

Once you've trained a volume, you can export the results as follows:

TODO
