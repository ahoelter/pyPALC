# Introduction

## What PALC does

A line source array is a vertical column of loudspeaker cabinets whose
tilt angles (the curving of the array) decide how evenly the sound reaches
the audience. PALC (Polygonal Audience Line Curving) computes these angles
analytically for a venue that is given as a slice through the audience
area: a polygon of straight audience lines.

pyPALC

* builds a continuous polygonal audience line from the audience lines
  (the venue proposal, which processes the gaps between them),
* computes the tilt angles of the cabinets with one of three PALC variants,
  optionally with discrete inter-cabinet angles, a fixed angle of the
  uppermost cabinet, a handling of the gaps between audience lines and a
  weighting that shapes the level over distance (linear, logarithmic or a
  target slope),
* predicts the sound field of the optimized array and of a reference array
  (straight, progressive, arc or user defined): frequency responses at the
  audience positions, the level over distance, the homogeneity over
  frequency and the level in the vertical radiation plane (beam plots).

The calculation (`pypalc.core`) has no GUI dependencies. It is used by the
command line runner (`pypalc run`, see {doc}`command-line`), by your own
scripts (see {doc}`tutorial`) and by the GUI (`pypalc gui`, see
{doc}`gui`); all three compute a setup in the same way and give the same
results.

## Browser app

Browser app: the address of the public pyPALC server follows here once it
is online. *(Placeholder: no public server yet.)*

## Publications

pyPALC is based on the following publications:

1. Straube, F.; Schultz, F.; Bonillo, D.A.; Weinzierl, S. (2018): “An
   Analytical Approach for Optimizing the Curving of Line Source Arrays.”
   In: *J. Audio Eng. Soc.* 66(1/2), 4–20.
2. Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2020): “Eine
   Web-Applikation zu Optimierung der Krümmung von Line Source Arrays.”
   In: *Fortschritte der Akustik: Tagungsband d. 46. DAGA*, Hannover.
3. Hölter, A.; Straube, F.; Schultz, F.; Weinzierl, S. (2021): “Enhanced
   Polygonal Audience Line Curving for Line Source Arrays.” In: *Proc. of
   the 149th Audio Eng. Soc. Conv.*, Online.
4. Hölter, A.; Schultz, F.; Straube F.; Weinzierl, S. (2021):
   “SPL-basierte Optimierung der Krümmung von Line-Source-Arrays mit PALC.”
   In: *Fortschritte der Akustik: Tagungsband d. 47. DAGA*, Wien.

## Authors

pyPALC is programmed and developed by Arne Hölter and Florian Straube
(Audio Communication Group, TU Berlin). Correspondence:
[hoelter@campus.tu-berlin.de](mailto:hoelter@campus.tu-berlin.de).

The source code is on GitHub:
[github.com/ahoelter/pyPALC](https://github.com/ahoelter/pyPALC).

## License

pyPALC is released under the MIT License (file `LICENSE` in the
repository).
