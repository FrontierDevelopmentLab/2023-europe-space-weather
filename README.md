# ICARUS


## Development Instructions

### Setup environment

#### Todo
```bash
conda create -f environment.yml
```

### Install pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will perform various checks before you are allowed to push to the repository. The default setup here will (among other things), check syntax, format code automatically and tidy import statements, remove output from notebooks (to save space) and prevent checking in of large files.

It's normal for pre-commit to fail if it fixes something. Usually the problem will be fixed, and you need to re-add the modified files before trying to commit again.
