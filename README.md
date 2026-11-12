# pyPALC

pyPALC optimizes the curving of line source arrays with PALC (Polygonal
Audience Line Curving): it computes the tilt angles of the loudspeaker
cabinets of an array for a given venue and predicts the sound field of the
curved array and of a reference array. It is a Python package with a
command line runner and a [Panel](https://panel.holoviz.org) GUI (with
[Bokeh](https://bokeh.org) figures) that runs in the browser. The
application is based on several scientific publications, which are listed
in the documentation and on the Help page of the GUI.

**Documentation: <https://pypalc.readthedocs.io>**

## Install

pyPALC needs Python 3.12 or newer.

```console
pip install pypalc
```

This installs the calculation, the command line runner (`pypalc`) and the
GUI (`pypalc gui` or `pypalc_gui`).

To work on pyPALC, clone the repository and install the pinned
dependencies of `requirements.txt` (conda) or `requirements_pip.txt` (pip);
see [Installation](https://pypalc.readthedocs.io/en/latest/installation.html)
and [Development](https://pypalc.readthedocs.io/en/latest/development.html).

## Run the GUI

```console
pypalc gui
```

(or `pypalc_gui`, which takes the same options) opens the app at
`http://localhost:5006/pyPALC` (options `--port`,
`--address`, `--allow-websocket-origin`, `--prefix`, `--no-show`). From a
checkout, without installing pyPALC, run `panel serve pyPALC --show` from the
folder that contains the `pyPALC` folder.

## Run a setup file without the GUI

```console
pypalc examples my_examples
pypalc run my_examples/simple.yaml --out results
```

computes the setup file and writes the tilt angles, the level over
distance, the homogeneity and the sound field as CSV files into `results`.
The GUI saves its input as such a setup file ("Save Setup"). See
[Command line](https://pypalc.readthedocs.io/en/latest/command-line.html)
and [Setup file reference](https://pypalc.readthedocs.io/en/latest/setup-file.html).

## License

MIT License, see [LICENSE](https://github.com/ahoelter/pyPALC/blob/master/LICENSE). Authors: Arne Hölter and Florian
Straube.
