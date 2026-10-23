#!/usr/bin/env python3
"""Plan the CI test shards and verify that every test ran exactly once.

plan:   Reads the XCTest and Swift Testing lists from `swift test list` and
        the measured durations in test-shards.json. Writes one plan per shard.
        Fails when a listed test is in no shard or in more than one shard.
verify: Reads the plans and the logs of all shards. Fails when the shards
        made different plans, or when a listed test did not run, ran two or
        more times, failed, or skipped without permission.

Units of work:
- Each XCTest class is one unit.
- A "Class/method" key in "seconds" makes that method its own unit.
- The isolated classes (fresh-process cap test and ring tests) are one unit.
  Their tests run in their own processes with their own environment.
- All Swift Testing tests are one unit. They run in parallel in one process.
The planner gives each unit, longest first, to the shard with the least work.
"""

import argparse
import json
import os
import re
import sys

XCTEST_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*/[A-Za-z_][A-Za-z0-9_]*$")
XCTEST_RESULT = re.compile(
    r"^Test Case '-\[([A-Za-z0-9_]+\.[A-Za-z0-9_]+) ([A-Za-z0-9_]+)\]' (passed|failed|skipped) \("
)
SWIFT_TESTING_RESULT = re.compile(r"^\S+ Test (?!run with )(.+) (passed|failed|skipped)( after [0-9.]+ seconds)?\.$")
SWIFT_TESTING_SUMMARY = re.compile(r"^\S+ Test run with ([0-9]+) tests? in [0-9]+ suites? (passed|failed)")

# Only these XCTest cases may skip. The perf test is opt-in. The retract test runs only on NAX hardware.
ALLOWED_SKIPS = {
    "QwenExpertTilePerfTests/testQwenRoutedPrefillUnitTimings",
    "SortedGatherQuantizedMMTests/testQwenSortednessViolationRetractsToLegacyRoute",
}

ISOLATED = "@isolated"
SWIFT_TESTING = "@swift-testing"


def fail(errors):
    for error in errors:
        print(f"::error::{error}")
    sys.exit(1)


def read_list(path, pattern=None):
    """Read one test identifier per line. Fail on a line that does not match the pattern."""
    tests = []
    errors = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if pattern is not None and not pattern.match(line):
                errors.append(f"{path}: the line '{line}' is not an XCTest identifier.")
            tests.append(line)
    if errors:
        fail(errors)
    return tests


def class_of(test):
    return test.split("/", 1)[0]


def short_id(test):
    """MLXTests.Class/method -> Class/method."""
    return test.split(".", 1)[1]


def plan(args):
    with open(args.config) as handle:
        config = json.load(handle)
    xctests = read_list(args.xctest, XCTEST_ID)
    swift_tests = read_list(args.swift_testing)
    errors = []

    if not xctests:
        errors.append("swift test list found no XCTest tests.")
    if len(set(xctests)) != len(xctests):
        errors.append("swift test list printed an XCTest identifier two times.")

    seconds = config["seconds"]
    default = float(config["default_seconds"])
    isolated_classes = set(config["isolated"]["classes"])
    listed_classes = {class_of(test) for test in xctests}
    for name in sorted(isolated_classes - listed_classes):
        errors.append(f"test-shards.json names the isolated class {name}, but swift test list has no such class.")
    for key in sorted(seconds):
        found = key in xctests if "/" in key else key in listed_classes
        if not found:
            # A stale duration does not change which tests run, so it is a warning.
            print(f"::warning::test-shards.json has a duration for {key}, but swift test list has no such test or class.")
    if errors:
        fail(errors)

    # Build the units. A unit is (name, seconds, tests).
    units = []
    split_methods = {key for key in seconds if "/" in key}
    by_class = {}
    for test in xctests:
        by_class.setdefault(class_of(test), []).append(test)
    for name, tests in sorted(by_class.items()):
        if name in isolated_classes:
            continue
        own = [test for test in tests if test in split_methods]
        rest = [test for test in tests if test not in split_methods]
        for test in own:
            units.append((test, float(seconds[test]), [test]))
        if rest:
            total = float(seconds.get(name, default)) - sum(float(seconds[test]) for test in own)
            units.append((name, max(total, default), rest))
    units.append((ISOLATED, float(config["isolated"]["seconds"]), []))
    if swift_tests:
        units.append((SWIFT_TESTING, float(config["swift_testing_seconds"]), []))

    # Longest first. Ties go by name, so the plan does not change from run to run.
    count = int(config["shards"])
    shards = [{"seconds": 0.0, "units": [], "xctest": []} for _ in range(count)]
    for name, cost, tests in sorted(units, key=lambda unit: (-unit[1], unit[0])):
        shard = min(range(count), key=lambda index: (shards[index]["seconds"], index))
        shards[shard]["seconds"] += cost
        shards[shard]["units"].append(name)
        shards[shard]["xctest"].extend(tests)

    # The guard: every listed test is in exactly one shard.
    owners = {}
    for index, shard in enumerate(shards, start=1):
        for test in shard["xctest"]:
            owners.setdefault(test, []).append(index)
        if ISOLATED in shard["units"]:
            for test in xctests:
                if class_of(test) in isolated_classes:
                    owners.setdefault(test, []).append(index)
    for test in xctests:
        found = owners.get(test, [])
        if not found:
            errors.append(f"The test {test} is in no shard.")
        elif len(found) > 1:
            errors.append(f"The test {test} is in shards {found}.")
    for test in owners:
        if test not in xctests:
            errors.append(f"The shard plan has {test}, but swift test list does not.")
    swift_owners = [index for index, shard in enumerate(shards, start=1) if SWIFT_TESTING in shard["units"]]
    if swift_tests and len(swift_owners) != 1:
        errors.append(f"The Swift Testing tests must be in exactly one shard, not in {swift_owners}.")
    if errors:
        fail(errors)

    os.makedirs(args.out, exist_ok=True)
    for index, shard in enumerate(shards, start=1):
        record = {
            "shard": index,
            "estimated_seconds": round(shard["seconds"], 1),
            "units": shard["units"],
            "xctest": shard["xctest"],
            "isolated": ISOLATED in shard["units"],
            "swift_testing": SWIFT_TESTING in shard["units"],
        }
        with open(os.path.join(args.out, f"shard-{index}.json"), "w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        # swift test --filter takes a regular expression. It matches Module.Class/method.
        regex = "^(" + "|".join(re.escape(test) for test in shard["xctest"]) + ")$" if shard["xctest"] else ""
        with open(os.path.join(args.out, f"shard-{index}.regex"), "w") as handle:
            handle.write(regex)
    with open(os.path.join(args.out, "xctest-tests.txt"), "w") as handle:
        handle.write("\n".join(xctests) + "\n")
    with open(os.path.join(args.out, "swift-testing-tests.txt"), "w") as handle:
        handle.write("\n".join(swift_tests) + ("\n" if swift_tests else ""))

    lines = [
        "### Test shards",
        "",
        f"{len(xctests)} XCTest tests and {len(swift_tests)} Swift Testing tests in {count} shards. Every test is in exactly one shard.",
        "",
        "| Shard | Estimated seconds | XCTest tests | Isolated processes | Swift Testing | Largest units |",
        "|---|---|---|---|---|---|",
    ]
    for index, shard in enumerate(shards, start=1):
        largest = ", ".join(f"`{unit}`" for unit in shard["units"][:3])
        lines.append(
            f"| {index} | {shard['seconds']:.0f} | {len(shard['xctest'])} | "
            f"{'yes' if ISOLATED in shard['units'] else 'no'} | "
            f"{'yes' if SWIFT_TESTING in shard['units'] else 'no'} | {largest} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    # Every shard makes the same plan. Only one shard writes it to the job summary.
    if summary and args.summary_shard in (None, 1):
        with open(summary, "a") as handle:
            handle.write(text)


def xctest_results(path):
    results = []
    if not os.path.exists(path):
        return results
    with open(path, errors="replace") as handle:
        for line in handle:
            match = XCTEST_RESULT.match(line.rstrip("\n"))
            if match:
                results.append((f"{match.group(1)}/{match.group(2)}", match.group(3)))
    return results


def verify(args):
    with open(args.config) as handle:
        config = json.load(handle)
    isolated_classes = set(config["isolated"]["classes"])
    count = int(config["shards"])
    # Each shard uploads its plan under plan/. The plan of shard 1 is the reference.
    reference = os.path.join(args.results, "test-results-1", "plan")
    xctests = read_list(os.path.join(reference, "xctest-tests.txt"), XCTEST_ID)
    swift_tests = read_list(os.path.join(reference, "swift-testing-tests.txt"))
    errors = []
    names = ["xctest-tests.txt", "swift-testing-tests.txt"] + [f"shard-{index}.json" for index in range(1, count + 1)]
    for index in range(2, count + 1):
        folder = os.path.join(args.results, f"test-results-{index}", "plan")
        for name in names:
            mine = os.path.join(folder, name)
            if not os.path.exists(mine):
                errors.append(f"Shard {index} has no plan file {name}.")
                continue
            with open(mine) as left, open(os.path.join(reference, name)) as right:
                if left.read() != right.read():
                    errors.append(f"Shard {index} made a different {name} than shard 1.")

    runs = {test: [] for test in xctests}
    ring_runs = {}
    swift_logs = []
    for index in range(1, count + 1):
        with open(os.path.join(reference, f"shard-{index}.json")) as handle:
            shard = json.load(handle)
        folder = os.path.join(args.results, f"test-results-{index}")
        if not os.path.isdir(folder):
            errors.append(f"Shard {index} uploaded no results.")
            continue
        logs = [("xctest.log", "xctest")] if shard["xctest"] else []
        if shard["isolated"]:
            logs.append(("cap.log", "xctest"))
            logs += [(name, "ring") for name in sorted(os.listdir(folder)) if name.startswith("ring-")]
        for name, kind in logs:
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                errors.append(f"Shard {index} has no {name}.")
                continue
            for test, status in xctest_results(path):
                if test not in runs:
                    errors.append(f"Shard {index} ran {test}, but swift test list does not have it.")
                    continue
                if kind == "ring":
                    # Each ring test starts its other ranks from the same bundle, and they print the same test case.
                    ring_runs.setdefault(test, []).append((index, status))
                else:
                    runs[test].append((index, status))
        if shard["swift_testing"]:
            swift_logs.append((index, os.path.join(folder, "swift-testing.log")))

    for test in xctests:
        if class_of(test) in isolated_classes and test in ring_runs:
            statuses = {status for _, status in ring_runs[test]}
            if statuses != {"passed"} or runs[test]:
                errors.append(f"{test} must pass in every rank, with no other run. Results: {ring_runs[test] + runs[test]}.")
            continue
        found = runs[test]
        if len(found) != 1:
            errors.append(f"{test} ran {len(found)} times, not exactly once: {found}.")
            continue
        status = found[0][1]
        if status == "failed":
            errors.append(f"{test} failed.")
        elif status == "skipped" and short_id(test) not in ALLOWED_SKIPS:
            errors.append(f"{test} skipped, and it does not have permission to skip.")

    if swift_tests:
        if len(swift_logs) != 1:
            errors.append(f"The Swift Testing tests must run in exactly one shard, not in {len(swift_logs)}.")
        for index, path in swift_logs:
            if not os.path.exists(path):
                errors.append(f"Shard {index} has no swift-testing.log.")
                continue
            results = []
            summary = None
            with open(path, errors="replace") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    match = SWIFT_TESTING_SUMMARY.match(line)
                    if match:
                        summary = (int(match.group(1)), match.group(2))
                        continue
                    match = SWIFT_TESTING_RESULT.match(line)
                    if match:
                        results.append((match.group(1), match.group(2)))
            if summary is None:
                errors.append(f"Shard {index}: the Swift Testing run printed no summary line.")
                continue
            if summary[0] != len(swift_tests):
                errors.append(
                    f"Shard {index}: Swift Testing ran {summary[0]} tests, but swift test list has {len(swift_tests)}."
                )
            passed = sum(1 for _, status in results if status == "passed")
            others = [result for result in results if result[1] != "passed"]
            if summary[1] != "passed" or others or passed != len(swift_tests):
                errors.append(
                    f"Shard {index}: {passed} of {len(swift_tests)} Swift Testing tests passed. Other results: {others[:20]}."
                )

    executed = sum(len(found) for found in runs.values())
    text = (
        "### Every test ran exactly once\n\n"
        f"XCTest: {len(xctests)} listed, {executed} results outside the ring ranks, "
        f"{len(ring_runs)} ring tests. Swift Testing: {len(swift_tests)} listed.\n"
    )
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as handle:
            handle.write(text if not errors else text + f"\n{len(errors)} errors. See the job log.\n")
    if errors:
        fail(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--xctest", required=True)
    plan_parser.add_argument("--swift-testing", required=True)
    plan_parser.add_argument("--out", required=True)
    plan_parser.add_argument("--summary-shard", type=int)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--results", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
