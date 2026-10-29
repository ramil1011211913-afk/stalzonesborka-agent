"""The release decision must fail on individual regressions or missing evidence."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_gate", Path(__file__).parents[1] / "release_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def evidence(tmp_path):
    pinned = tmp_path / "source.py"
    pinned.write_text("source")
    memory = dict(sampling_errors=[], samples=1, peak_rss_bytes=100, sampled_pool_used_bytes=100)
    return dict(baseline=gate.BASELINE, sha256={"source.py": gate.digest(pinned)},
                configuration=copy.deepcopy(gate.CANDIDATE), test_count=10,
                tests=[dict(name=n, returncode=0, passed=10, failed=0, skipped=0,
                            actual_gpu=n.startswith("gpu")) for n in sorted(gate.MATRIX)],
                case_ids=["case"], comparisons=[dict(case="case", backend=b, results={
                    s: dict(key=[-10, 100], invalid=0, checked=1) for s in gate.STRATEGIES}) for b in gate.BACKENDS],
                oracle_count=1, oracle=[dict(id="oracle", passed=True)],
                milp_count=1, milp=[dict(id="model", certified=True, candidate_key=[-10, 100], optimum_key=[-10, 100])],
                resource_policy=dict(approved=True, kind="relative", median_ratio=1.25, memory_ratio=1.1),
                minimum_repetitions=3, performance_cases=["case"], performance=[
                    dict(case="case", backend=b, strategy=s, times=[1, 1, 1], memory=[memory.copy()])
                    for b in gate.BACKENDS for s in gate.STRATEGIES],
                preservation=dict(head=gate.BASELINE, branch="experiment/beam-danger-lookahead", staged=0,
                                  index_unchanged=True, changed_files=[], checked_files=1))


def test_complete_evidence_can_pass(evidence, tmp_path):
    result = gate.evaluate(evidence, tmp_path)
    assert result["decision"] == "GO" and result["completion_percent"] == 100


@pytest.mark.parametrize("section", ["sha256", "configuration", "tests", "comparisons", "oracle", "milp", "resource_policy", "performance", "preservation"])
def test_missing_evidence_fails_closed(evidence, tmp_path, section):
    del evidence[section]
    assert gate.evaluate(evidence, tmp_path)["decision"] == "NO-GO"


@pytest.mark.parametrize("key", [[-9, 1], [-10, 101], [float("nan"), 0], [float("inf"), 0], None])
def test_one_bad_rank_blocks_release(evidence, tmp_path, key):
    evidence["comparisons"][0]["results"]["candidate"]["key"] = key
    assert "no_full_rank_regressions" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_better_primary_can_cost_more():
    assert gate.no_worse([-11, 1000], [-10, 100])
    assert not gate.no_worse([-10, 101], [-10, 100])
    assert not gate.no_worse([-10, -1, 1], [-10, -2, 100])


def test_duplicate_case_does_not_replace_missing_backend(evidence, tmp_path):
    evidence["comparisons"][1] = copy.deepcopy(evidence["comparisons"][0])
    assert "no_full_rank_regressions" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_resource_outlier_is_not_averaged_away(evidence, tmp_path):
    evidence["performance"][0]["strategy"] = "candidate"  # duplicates or bad memory both block
    evidence["performance"][0]["memory"][0]["peak_rss_bytes"] = 1000
    assert "approved_resource_limits" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_source_change_invalidates_evidence(evidence, tmp_path):
    (tmp_path / "source.py").write_text("changed")
    assert "pinned_sources_and_evidence" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_unapproved_limits_do_not_become_approval(evidence, tmp_path):
    evidence["resource_policy"]["approved"] = False
    assert "approved_resource_limits" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_wrong_backend_cannot_claim_gpu_coverage(evidence, tmp_path):
    evidence["tests"][0]["actual_gpu"] = not evidence["tests"][0]["actual_gpu"]
    assert "four_regression_modes" in gate.evaluate(evidence, tmp_path)["blockers"]


def test_no_evidence_is_not_ready(tmp_path):
    result = gate.evaluate({}, tmp_path)
    assert result["decision"] == "NO-GO" and result["completion_percent"] == 0
