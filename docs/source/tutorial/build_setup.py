"""Build a setup in code, save it as a setup file and compute it."""
import dataclasses

from pypalc.core import pipeline
from pypalc.core.errors import SetupError
from pypalc.core.settings import (AlgorithmSettings, ArraySettings,
                                  AudienceLine, LoudspeakerSettings,
                                  ReferenceArraySettings, Setup,
                                  VenueSettings, dump_setup, load_setup,
                                  save_setup, validate)

# [build]
# Every setting that is not given has its default value.
setup = Setup(
    venue=VenueSettings(audience_lines=(
        AudienceLine(start=(3.0, 0.0), stop=(12.0, 1.5),
                     points_per_meter=4.0),
        AudienceLine(start=(14.0, 2.5), stop=(25.0, 5.0),
                     points_per_meter=4.0),
    )),
    loudspeaker=LoudspeakerSettings(cabinet_height=0.25),
    array=ArraySettings(cabinet_count=8, top_position=(0.0, 6.0),
                        cabinet_gap=0.01),
    reference_array=ReferenceArraySettings(type="arc", top_angle_deg=0.0,
                                           angle_param_deg=3.0),
    algorithm=AlgorithmSettings(variant="palc2", gap_handling="soft_margin",
                                soft_margin_strength=1.2, tolerance=1.0),
)
# [validate]
# The dataclasses do not check their values; validate() does, as loading a
# setup file does.
validate(setup)
try:
    validate(dataclasses.replace(
        setup, algorithm=dataclasses.replace(setup.algorithm, tolerance=10)))
except SetupError as exc:
    print("invalid:", exc.key, "-", exc.message)
# [save]
print(dump_setup(setup))            # the YAML text
save_setup(setup, "my_setup.yaml")  # the same text as a file
assert load_setup("my_setup.yaml") == setup
# [run]
result = pipeline.run(setup)
print("tilt angles in deg:", result.optimized.tilt_angles_deg)
print("reference tilt angles in deg:", result.reference.tilt_angles_deg)

# The settings are frozen dataclasses: dataclasses.replace() makes a
# changed copy.
palc3 = dataclasses.replace(
    setup, algorithm=dataclasses.replace(setup.algorithm, variant="palc3"))
print("PALC 3 tilt angles in deg:",
      pipeline.run(palc3).optimized.tilt_angles_deg)
