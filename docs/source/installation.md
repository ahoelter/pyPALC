# Installation

pyPALC needs Python 3.12 or newer.

## From PyPI

The calculation and the command line runner, with numpy, scipy and PyYAML
as their only dependencies:

```console
pip install pypalc
```

With the GUI (Panel, panel-material-ui, Bokeh and the other packages the
GUI imports):

```console
pip install "pypalc[gui]"
```

The quotes keep the shell from reading the brackets. Check the
installation with `pypalc --help`; `pypalc gui` starts the GUI (see
{doc}`gui`).

Further extras: `docs` (Sphinx, Furo and MyST-Parser, to build this
documentation), `test` (pytest) and `dev` (build and twine, to build and
upload a release).

## From source

Clone the repository:

```console
git clone https://github.com/ahoelter/pyPALC.git
cd pyPALC
```

The files `requirements.txt` and `requirements_pip.txt` pin the versions
pyPALC is developed and tested with (Python 3.12, Panel 1.10,
panel-material-ui 0.16, Bokeh 3.10 and pytest). Install them in one of two
ways.

With conda, from the conda-forge channel:

```console
conda create -n pyPALC3 -c conda-forge --file requirements.txt
conda activate pyPALC3
```

An environment made for an earlier version of pyPALC has no Panel; create
it anew.

With pip, in a Python 3.12 environment (the pinned versions are the same
as in `requirements.txt`):

```console
pip install -r requirements_pip.txt
```

Then install pyPALC itself in editable mode, which adds the `pypalc`
command and keeps the package pointing at the checkout:

```console
pip install -e .
```

A checkout also runs the GUI without installing it:
`panel serve pyPALC --show`, from the folder that contains the `pyPALC`
folder (see {doc}`gui`).

## Packages in `~/.local`

Packages installed with `pip install --user` (in `~/.local`) shadow those
of a conda or virtual environment. Set `PYTHONNOUSERSITE=1` for every
command to ignore them, for example

```console
PYTHONNOUSERSITE=1 conda run --no-capture-output -n pyPALC3 panel serve pyPALC --show
```

(without `--no-capture-output`, `conda run` holds back the server log until
the server stops), or activate the environment first:

```console
conda activate pyPALC3
PYTHONNOUSERSITE=1 panel serve pyPALC --show
```
