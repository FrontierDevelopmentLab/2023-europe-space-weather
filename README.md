# ICARUS


## Development Instructions

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