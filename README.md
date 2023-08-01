# ICARUS

## Getting Started

### Getting started with the onboard component Onboard - Overview


The onboard component of the project pertains to improvements of onboard compression algorithms and automated detection of CMEs onboard. This will enable a smaller overhead in terms of time from detection of a CME to the possibility of reacting to it, as well as possibly enabling the collection of further data around the event. This compressed data then may also inform the groundbased components of the system, creating a 3D reconstruction of a CME. The target of this project is the future VIGIL spacecraft, set to be launched by the ESA in 2029. We are to show the viability of onboard-machine learning to enhance the runtime environment, possibly with a dual system to collate different datasets onground. The Aim is thus to
    - Increase the data rate for transmission from the satellite
    - Showcase the usefulness of ML algorithms for onboard usage
    - Generate the first ever 3D reconstruction of a CME

The onboard component has a "tutorials" subfolder. In that folder, there are currently 2 files, "compressAI_experiments.ipynb" and "Onboard_EDA.ipynb".
The main files in the onboard folder are what is actually used to run the program. Download_tools provides a download to the [ICER Instrument Suite](https://stereo-ssc.nascom.nasa.gov/instruments/software/secchi/utils/icer/), though this is lacking usefulness due to the inability to run the compression algorithm locally ([ICER](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155J.pdf) has initially been considered the primary compression algorithm used, though in the absence of viable metrics, we are now looking at the [J2K algorithm](https://jpeg.org/jpeg2000/index.html)).

To get started (once ephemeris files are downloaded via the system discussed further in the onboard_eda subsection), use the "download_satellite_data.py" program to download the required data for the Corona 1 and Corona 2 instruments from the STEREO A/B spacecraft.
Once this is done, the data_loader.py is useable through the compressAI_Inference file to load the downloaded data and compress it, using a pretrained model loaded with the compressAI library to generate example compression through VQ-VAEs. Additionally, the loaded events enable the marking of a measurement with the presence of a CME - enabling the training of binary detection mechanisms.

#### Onboard_EDA.ipynb

The Onboard EDA notebook contains information on how data is sampled from the STEREO A/B satellites in a physically relevant way. The program takes .ephemeris files from the [Nasa Horizons Ephemeris tool](https://ssd.jpl.nasa.gov/horizons/app.html#/), which can be set up to yield location data for specified spacecraft or points of interest in a given reference frame. For this task, we have set the tool to query the locations of L5, SOHO, Stereo A/B on a daily basis in a heliocentral coordinate system (IE, the Sun is at (0,0,0)). The Ephemeris files report the Right Ascension (R.A.) and Declination (Dec), alongside radius r and rate of change of radius w.r.t. time $\dot{r}$.
Given that the inner angle from the sun in the L5-Sun-Earth triangle is 60°, we can specify L5 like scenarios as those where the satellites take on a similar 60° inner angle.
In the notebook and actual code, these times are computed that way from the position data.
Based on how the timeseries changes, we can thus extract batches of time when relevant scenarios are covered.

The Sunpy library has access to FIDO, a system for querying satellite data and known heliophysical events. Using the computed batches of relevant points in time, the notebook can report the data that should be considered by the program. In the actual program, the reported data and events are saved for future use - Data is saved in the fits format (as is the download), events are saved by noting their range of activity, then saving that in an event filestructure under the data folder noted in the configuration file (onboard.yaml).
Finally, the notebook is set to create a visualization of the data. This can be extrapolated for the 3D setup explanation possibly required for some publications.

#### compressAI_experiments.ipynb

This file is based on the [example notebook found on the compressAI Github] (https://github.com/InterDigitalInc/CompressAI/blob/master/examples/CompressAI%20Inference%20Demo.ipynb). It shows the usage of the library to load compression models, taking image data and compressing it into latentspace. It also denotes metrics to use with compression algorithms, showcasing the viability of the chosen approach to compression through them. However, the code here is not generalized into our project. The version that does this is found in the "compressAI_Inference.py" file.


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
### Install OpenJPEG2000
The onboard component of the code uses glymur, a library meant for JPEG2000 compression to generate comparisons. To install this, the library for OpenJPEG is going to be needed.
In order to get this set up, you need to
```bash
sudo apt install python3-glymur
```
which will install the C library [from here](https://github.com/uclouvain/openjpeg) and configure it correctly.

### Install `icarus`

With the env installed, you can install the module:

```bash
pip install -e .[onboard,test]

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
