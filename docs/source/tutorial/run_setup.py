"""Run the example setup ``simple.yaml`` from Python and write its results."""
from pathlib import Path

from pypalc.core import pipeline
from pypalc.core.export import write_results
from pypalc.core.settings import load_setup
from pypalc.examples import copy_examples, load_example

# [load]
# An example that ships with pyPALC, by its name (the file name without
# ".yaml"):
setup = load_example("simple")

# Any setup file: here a copy of the examples in the folder "my_examples"
# (as "pypalc examples my_examples" makes it).
folder = Path("my_examples")
if not (folder / "simple.yaml").exists():
    copy_examples(folder)
setup = load_setup(folder / "simple.yaml")
print("cabinets:", setup.array.cabinet_count)
# [run]
result = pipeline.run(setup)

# the optimized array
print("tilt angles in deg:", result.optimized.tilt_angles_deg)
print("inter-cabinet angles in deg:",
      result.optimized.inter_cabinet_angles_deg)
print("PALC iterations:", result.geometry.iteration_count)
# the reference array
print("reference tilt angles in deg:", result.reference.tilt_angles_deg)

# the sound field prediction of the optimized array
evaluation = result.optimized.evaluation
print("frequencies:", len(evaluation.frequencies))
print("SPL at the audience positions:", evaluation.spl_db.shape)
# [arrays]
arrays = pipeline.result_arrays(result)
for name, values in arrays.items():
    print("%-32s %s" % (name, values.shape))

# mean homogeneity (spread of the levels over the audience positions; the
# smaller, the more even) from 200 Hz to 8 kHz, PALC against the reference
frequencies = arrays["frequencies_hz"]
band = (frequencies >= 200) & (frequencies <= 8000)
print("homogeneity, 200 Hz to 8 kHz: PALC %.1f dB, reference %.1f dB"
      % (arrays["homogeneity_db"][band].mean(),
         arrays["reference_homogeneity_db"][band].mean()))
# [write]
for path in write_results(result, Path("results")):
    print("written:", path)
