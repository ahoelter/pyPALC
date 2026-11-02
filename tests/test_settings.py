"""Tests for setup settings loading, saving and validation."""
import pytest

from pypalc.core.errors import SetupError
from pypalc.core.settings import (
    AlgorithmSettings, ArraySettings, AudienceLine, LoudspeakerSettings,
    MeasuredDirectivity, ReferenceArraySettings, Setup, TargetPoint,
    TargetSlopeSettings, VenueSettings, dump_setup, load_setup,
    load_setup_text,
)


def test_defaults_round_trip():
    assert load_setup_text(dump_setup(Setup())) == Setup()


def test_full_round_trip():
    setup = Setup(
        venue=VenueSettings(audience_lines=(
            AudienceLine((5.0, 1.0), (20.0, 0.5), 2.0),
            AudienceLine((22.0, 1.0), (40.0, 1.5)),
        )),
        loudspeaker=LoudspeakerSettings(
            directivity="measured",
            measured_directivity=MeasuredDirectivity(csv="a,b\n1,2\n3,4\n")),
        array=ArraySettings(cabinet_count=8,
                            discrete_tilt_angles_deg=(0.0, 1.0, 2.5)),
        reference_array=ReferenceArraySettings(
            type="arc", top_angle_deg=1.0, angle_param_deg=0.5,
            use_discrete_angles=True),
        algorithm=AlgorithmSettings(
            gap_handling="soft_margin", soft_margin_strength=0.5,
            weighting="target_slope", fixed_top_angle_deg=2.5),
        target_slope=TargetSlopeSettings(
            points=(TargetPoint(5.0, 0.0), TargetPoint(15.0, -3.0),
                    TargetPoint(30.0, -8.0)),
            reference_distance=10.0),
    )
    assert load_setup_text(dump_setup(setup)) == setup


def _key(text, **kw):
    with pytest.raises(SetupError) as exc:
        load_setup_text(text, **kw)
    return exc.value.key


def test_unknown_key_names_key():
    assert _key("array: {cabinets: 4}") == "array.cabinets"


def test_wrong_type_names_key():
    assert _key("array: {cabinet_count: four}") == "array.cabinet_count"


def test_int_accepted_as_float():
    s = load_setup_text("algorithm: {tolerance: 1}")
    assert s.algorithm.tolerance == 1.0 and isinstance(s.algorithm.tolerance, float)


@pytest.mark.parametrize("section,field,value,key", [
    ("array", "cabinet_count", "0", "array.cabinet_count"),
    ("array", "cabinet_count", "31", "array.cabinet_count"),
    ("algorithm", "tolerance", "0", "algorithm.tolerance"),
    ("algorithm", "variant", "palc4", "algorithm.variant"),
])
def test_out_of_range(section, field, value, key):
    assert _key("%s: {%s: %s}" % (section, field, value)) == key


def test_target_slope_required():
    assert _key("algorithm: {weighting: target_slope}") == "target_slope"


def test_target_slope_with_hard_margin_rejected():
    text = ("algorithm: {weighting: target_slope, gap_handling: hard_margin}\n"
            "target_slope:\n"
            "  reference_distance: 10\n"
            "  points: [{distance: 5, level_db: 0}, {distance: 20, level_db: -3}]\n")
    assert _key(text) == "algorithm.gap_handling"


@pytest.mark.parametrize("md", [
    "{path: d.csv, csv: 'x'}",
    "{}",
])
def test_measured_requires_exactly_one_source(md):
    text = "loudspeaker: {directivity: measured, measured_directivity: %s}" % md
    assert _key(text, base_dir=None) == "loudspeaker.measured_directivity"


def test_relative_path_resolved_against_file(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "s.yaml"
    f.write_text("loudspeaker: {directivity: measured, "
                 "measured_directivity: {path: d.csv}}")
    s = load_setup(f)
    assert s.loudspeaker.measured_directivity.path == str(tmp_path / "sub" / "d.csv")


def test_text_load_rejects_path_without_base_dir():
    text = ("loudspeaker: {directivity: measured, "
            "measured_directivity: {path: d.csv}}")
    assert _key(text) == "loudspeaker.measured_directivity.path"


def test_version_mismatch():
    assert _key("version: 2") == "version"
