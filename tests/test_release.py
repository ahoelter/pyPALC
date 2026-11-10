"""Release tooling: tag check, publishing workflow, buildable distributions."""

import importlib.util
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
CHECK = ROOT / "tools" / "check_version_tag.py"


def _run_check(tag):
    return subprocess.run(
        [sys.executable, str(CHECK), tag], capture_output=True, text=True
    )


def _version():
    spec = importlib.util.spec_from_file_location("check_version_tag", CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_version()


def test_matching_tag_passes():
    assert _run_check("v" + _version()).returncode == 0


def test_mismatching_tag_fails_and_prints_both():
    good = "v" + _version()
    bad = good + ".post1"
    result = _run_check(bad)
    assert result.returncode == 1
    assert bad in result.stdout and good in result.stdout


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def test_workflow_triggers(workflow):
    on = workflow.get("on", workflow.get(True))  # YAML 1.1 reads `on` as True
    assert on["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in on


@pytest.mark.parametrize(
    "job, environment", [("publish-testpypi", "testpypi"), ("publish-pypi", "pypi")]
)
def test_publish_jobs(workflow, job, environment):
    spec = workflow["jobs"][job]
    assert spec["environment"] == environment
    assert spec["permissions"] == {"id-token": "write"}
    uses = [s.get("uses", "") for s in spec["steps"]]
    assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in uses)


def test_publish_job_conditions_and_artifact(workflow):
    jobs = workflow["jobs"]
    assert jobs["publish-testpypi"]["if"] == "github.event_name == 'workflow_dispatch'"
    assert jobs["publish-pypi"]["if"] == (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    )
    for name in ("publish-testpypi", "publish-pypi"):
        assert jobs[name]["needs"] == "build"
        downloads = [
            s for s in jobs[name]["steps"]
            if s.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert [d["with"]["name"] for d in downloads] == ["dist"]


def test_testpypi_uses_test_repository_url(workflow):
    steps = workflow["jobs"]["publish-testpypi"]["steps"]
    publish = [s for s in steps if "gh-action-pypi-publish" in s.get("uses", "")][0]
    assert publish["with"]["repository-url"] == "https://test.pypi.org/legacy/"
    steps = workflow["jobs"]["publish-pypi"]["steps"]
    publish = [s for s in steps if "gh-action-pypi-publish" in s.get("uses", "")][0]
    assert "repository-url" not in publish.get("with", {})


def test_workflow_has_no_secrets():
    assert "secrets." not in WORKFLOW.read_text()


def test_build_and_twine_check(tmp_path):
    if importlib.util.find_spec("build") is None:
        pytest.skip("the 'build' package is not installed")
    if importlib.util.find_spec("twine") is None:
        pytest.skip("the 'twine' package is not installed")
    out = tmp_path / "dist"
    cmd = [sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:  # build isolation needs the network
        result = subprocess.run(
            cmd + ["--no-isolation"], capture_output=True, text=True
        )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    files = sorted(out.iterdir())
    check = subprocess.run(
        [sys.executable, "-m", "twine", "check", *map(str, files)],
        capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    sdist = next(f for f in files if f.name.endswith(".tar.gz"))
    with tarfile.open(sdist) as tar:
        names = tar.getnames()
    assert any(n.endswith("pypalc/examples/simple.yaml") for n in names)
    assert any("pypalc/gui/static/" in n for n in names)
