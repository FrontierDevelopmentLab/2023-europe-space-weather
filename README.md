# ICARUS


## Development Instructions

### Setup environment

This doesn't exist yet.

```bash
conda create -f environment.yml
conda activate helio23
pip install -e .
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
