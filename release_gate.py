"""Offline release decision for the explicit chunked Quality 2000 candidate.

This tool never runs searches, stages files, changes defaults, or publishes a
release. Evidence is produced in an isolated audit; missing evidence fails closed.
Completion measures passed mandatory checks, not probability of optimality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


CANDIDATE = dict(beam_width=2000, danger_lookahead=True,
                 expansion_chunk_size=262144, constraint_lookahead=False,
                 joint_budget_tail=False, joint_tail_frontier=False,
                 portfolio=False, default_danger_lookahead=False)
BASELINE = "c785e231b91b28dd0214cc1b72cf2debbbfcdbe3"
MATRIX = {"gpu-0", "gpu-1", "cpu-0", "cpu-1"}
STRATEGIES = {"stable", "quality", "candidate"}
BACKENDS = {"cpu", "gpu"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_vector(value):
    return (isinstance(value, list) and len(value) >= 2 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value))


def no_worse(candidate, baseline):
    """Minimize sort keys: all objective components first, then exact price.

    Only roundoff (1e-9 absolute) is allowed on objective components. Prices
    must not worsen on a primary tie. Lexicographic bins are integral.
    """
    if baseline is None:
        return candidate is None or numeric_vector(candidate)
    if not numeric_vector(candidate) or not numeric_vector(baseline):
        return False
    if len(candidate) != len(baseline):
        return False
    for a, b in zip(candidate[:-1], baseline[:-1]):
        if a < b - 1e-9:
            return True
        if a > b + 1e-9:
            return False
    return candidate[-1] <= baseline[-1]


def evaluate(evidence, root):
    """Return every blocker; no aggregate quality score can mask a bad case."""
    root = Path(root).resolve()
    checks = []

    def check(name, predicate):
        try:
            ok = bool(predicate())
        except (KeyError, TypeError, ValueError, OSError, IndexError, ZeroDivisionError):
            ok = False
        checks.append(dict(name=name, passed=ok))

    def provenance():
        pins = evidence["sha256"]
        if not pins or evidence["baseline"] != BASELINE:
            return False
        for name, expected in pins.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or digest(path) != expected:
                return False
        return True

    check("pinned_sources_and_evidence", provenance)
    check("safe_explicit_configuration", lambda: evidence["configuration"] == CANDIDATE)

    def tests():
        rows = evidence["tests"]
        return (len(rows) == 4 and {r["name"] for r in rows} == MATRIX and
                all(r["returncode"] == 0 and r["failed"] == 0 and r["passed"] > 0
                    and r["passed"] + r["skipped"] == evidence["test_count"]
                    and r["actual_gpu"] == r["name"].startswith("gpu") for r in rows))
    check("four_regression_modes", tests)

    def comparisons():
        rows = evidence["comparisons"]
        expected = {(case, backend) for case in evidence["case_ids"] for backend in BACKENDS}
        actual = [(r["case"], r["backend"]) for r in rows]
        if not expected or len(actual) != len(expected) or set(actual) != expected:
            return None
        return rows

    check("all_returned_builds_feasible", lambda: comparisons() is not None and
          all(set(r["results"]) == STRATEGIES and
              all(v["invalid"] == 0 and v["checked"] > 0 and numeric_vector(v["key"])
                  for v in r["results"].values()) for r in comparisons()))
    check("no_full_rank_regressions", lambda: comparisons() is not None and
          all(no_worse(r["results"]["candidate"]["key"], r["results"][s]["key"])
              for r in comparisons() for s in ["stable", "quality"]))
    check("independent_exhaustive_controls", lambda:
          len(evidence["oracle"]) == evidence["oracle_count"] > 0 and
          len({r["id"] for r in evidence["oracle"]}) == evidence["oracle_count"] and
          all(r["passed"] is True for r in evidence["oracle"]))
    check("milp_controls_completed", lambda:
          len(evidence["milp"]) == evidence["milp_count"] > 0 and
          len({r["id"] for r in evidence["milp"]}) == evidence["milp_count"] and
          all(r["certified"] is True and numeric_vector(r["candidate_key"])
              and numeric_vector(r["optimum_key"]) for r in evidence["milp"]))

    def resources():
        policy = evidence["resource_policy"]
        if policy["approved"] is not True or policy["kind"] not in ("relative", "absolute"):
            return False
        rows = evidence["performance"]
        expected = {(case, backend, strategy) for case in evidence["performance_cases"]
                    for backend in BACKENDS for strategy in STRATEGIES}
        actual = [(r["case"], r["backend"], r["strategy"]) for r in rows]
        if not expected or len(actual) != len(expected) or set(actual) != expected:
            return False
        indexed = {(r["case"], r["backend"], r["strategy"]): r for r in rows}
        for r in rows:
            if (len(r["times"]) < evidence["minimum_repetitions"] or
                    any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in r["times"]) or
                    not r["memory"] or any(m["sampling_errors"] or m["samples"] < 1 for m in r["memory"])):
                return False
            if r["strategy"] != "candidate":
                continue
            q = indexed[r["case"], r["backend"], "quality"]
            median = statistics.median(r["times"])
            rss = max(m["peak_rss_bytes"] for m in r["memory"])
            used = max(m["sampled_pool_used_bytes"] for m in r["memory"])
            if policy["kind"] == "relative":
                limits = (statistics.median(q["times"]) * policy["median_ratio"],
                          max(m["peak_rss_bytes"] for m in q["memory"]) * policy["memory_ratio"],
                          max(m["sampled_pool_used_bytes"] for m in q["memory"]) * policy["memory_ratio"])
            else:
                limits = (policy[r["backend"] + "_seconds"], policy["rss_bytes"], policy["pool_used_bytes"])
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (rss, used, *limits)):
                return False
            if median > limits[0] or rss > limits[1] or (r["backend"] == "gpu" and used > limits[2]):
                return False
        return True
    check("approved_resource_limits", resources)
    check("git_and_existing_files_unchanged", lambda:
          evidence["preservation"]["head"] == BASELINE and
          evidence["preservation"]["branch"] == "experiment/beam-danger-lookahead" and
          evidence["preservation"]["staged"] == 0 and
          evidence["preservation"]["index_unchanged"] is True and
          evidence["preservation"]["changed_files"] == [] and
          evidence["preservation"]["checked_files"] > 0)
    passed = sum(c["passed"] for c in checks)
    return dict(decision="GO" if passed == len(checks) else "NO-GO",
                completion_percent=round(100 * passed / len(checks), 1),
                passed=passed, mandatory=len(checks), checks=checks,
                blockers=[c["name"] for c in checks if not c["passed"]],
                scope="Pinned offline suite only; no global optimality or deployment approval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="New report file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = evaluate(evidence, args.root)
    except (OSError, ValueError, TypeError) as error:
        result = dict(decision="NO-GO", completion_percent=0, blockers=[str(error)])
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
