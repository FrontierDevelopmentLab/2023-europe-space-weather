# ICARUS


## Development Instructions

### Notes

This setup is designed to work cleanly in a shared system. On your own machine, you can probably get away with using `conda`/`mamba` or a similar package manager. Instructions below assume a Linux VM, but once you've got `pyenv` installed they should work on any system.

### Install pyenv

First, get any system dependencies

```bash
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Then install pyenv:

```bash
curl https://pyenv.run | bash
```

Add the following to your `.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Disable Conda's auto-start:

```bash
conda config --set auto_activate_base false
```

Restart your shell, or source your `~/.bashrc` file and create your environment:

```bash
pyenv install 3.10
pyenv virtualenv 3.10 onboard
```

Change directory to here, and:

```bash
pyenv local onboard
```

This will automatically activate the `onboard` environment when you change to the repository folder.

### Install `icarus`

With the env installed, you can install the module:

```bash
pip install -e .[onboard,ttest]

# or

pip install -e .[ground,test]
```

Using `-e` will make an editable install so that changes to the package will be immediately reflected in your scripts/notebooks (when you restart the kernel).

### Install pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will perform various checks before you are allowed to push to the repository. The default setup here will (among other things), check syntax, format code automatically and tidy import statements, remove output from notebooks (to save space) and prevent checking in of large files.

It's normal for pre-commit to fail if it fixes something. Usually the problem will be fixed, and you need to re-add the modified files before trying to commit again.

### Run the test suite

```bash
pip install pytest pytest-cov
python -m pytest test
```

### Specify new dependencies

Dependencies are specified in the `pyproject` file. Only add dependencies which are required by the project to avoid bloated environments. Add _universal_ dependencies (like Pytorch) in the `[project]` section.

Add optional dependencies for the different toolsets in `[project.optional-dependencies]`.

## Set up training on ScanAI

### Log in to Scan

Download and install [openvpn](https://openvpn.net/client/).

Prepare `fdl.ovpn` config file.

Start openvpn client with the config file. (linux)
```bash
sudo openvpn --config fdl.ovpn
```

```bash
ssh fdl@172.18.2.20
```
Enter the password.

### Clone git repo

Once logged in, the username is `fdl` by default.

```bash
mkdir workspace
cd workspace
git clone https://github.com/FrontierDevelopmentLab/2023-europe-space-weather.git
```

### Download training data

Follow commands in `scripts/run_icarus.sh` to download data to `~/mnt` instead of `/mnt`.

Note: No permission to modify `/mnt`.

```bash
mkdir ~/mnt
mkdir ~/mnt/ground-data
gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/data_fits ~/mnt/ground-data/
gsutil -m cp -R gs://fdl23_europe_helio_onground/ground-data/PSI ~/mnt/ground-data/
gsutil -m cp -R gs://fdl_space_weather_data/events/fdl_stereo_2014_02_prep.zip ~/mnt/ground-data/
```

### Screen session (optional)

Start a screen session before running a docker container.
```bash
screen
```

### Docker

Download the [PyTorch Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
```bash
docker pull nvcr.io/nvidia/pytorch:22.01-py3
```

Run a Docker container from the image above and mount the appropriate volumes.
```bash
docker run -v /home/fdl/workspace/2023-europe-space-weather/:/workspace/2023-europe-space-weather/ -v /home/fdl/mnt:/mnt --gpus all -it --rm nvcr.io/nvidia/pytorch:22.01-py3
```

Inside the docker container, install the requirements.
```bash
pip install -r requirements.txt
```

### Data prep

Unzip the zip file.
```bash
python
```

```python
import zipfile

path_to_zip_file = "/mnt/ground-data/fdl_stereo_2014_02_prep.zip"
directory_to_extract_to = "/mnt/ground-data/data_fits_stereo_2014_02"

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
```

Follow commands in `scripts/run_icarus.sh` to prep data in `/mnt/prep-data`.
