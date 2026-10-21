#!/usr/bin/env python3
"""Check generated Metal text against nested core without a compiler.

Default scope: affine and floating-point quantized kernels, including NAX. --all audits every copied
Metal file and declared Metal JIT resource (not CPU/CUDA generated sources).
This reconstructs the generator's declared sections, not its compiler-derived
include discovery. Full regeneration remains a separate toolchain gate.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

DEFAULT_NAMES = ("quantized", "quantized_nax", "fp_quantized", "fp_quantized_nax")
KERNELS = "mlx/backend/metal/kernels/"
SEPARATOR = "/" * 79


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copied_text(source, relative):
    prefix = "../" * (len(PurePosixPath(relative).parts) - 1)
    return re.sub(r'^#include "mlx/backend/metal/kernels/([^\"]*)"',
                  lambda match: '#include "' + prefix + match[1] + '"',
                  source, flags=re.MULTILINE)


def expanded_text(source):
    # Same line filtering as make_compiled_preamble.sh; do not normalize whitespace.
    # grep emits a newline even when the final source line has none.
    if not source:
        return ""
    lines = source.split("\n")
    if source.endswith("\n"):
        lines.pop()
    return "".join(line + "\n" for line in lines
                   if not re.search(r'#include ".*.h"|#pragma once', line))


def jit_text(name, root, sections):
    content = "// Copyright © 2025 Apple Inc.\n\n// Auto generated source for " + root + "\n\n"
    for path, source in sections:
        content += (SEPARATOR + '\n// Contents from "' + path + '"\n' + SEPARATOR
                    + '\n\n#line 1 "' + path + '"\n' + expanded_text(source) + "\n")
    return ('namespace mlx::core::metal {\n\nconst char* ' + name
            + '() {\n  return R"preamble(\n' + content + SEPARATOR
            + '\n)preamble";\n}\n\n} // namespace mlx::core::metal\n')


def audit(package, names=DEFAULT_NAMES, all_sources=False):
    package = Path(package).resolve()
    core = package / "Source/Cmlx/mlx"
    generated = package / "Source/Cmlx/mlx-generated"
    hashes, errors, resources = {}, [], []

    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(package))] = sha(data)
        return data.decode("utf-8")

    def input_path(relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not relative.startswith(KERNELS):
            raise ValueError("non-core declared input: " + relative)
        result = core / relative
        if not result.resolve().is_relative_to(core.resolve()):
            raise ValueError("input escapes nested core: " + relative)
        return result

    if all_sources:
        headers = {p.relative_to(generated / "metal").as_posix()
                   for p in (generated / "metal").rglob("*") if p.is_file()}
        headers.update(p.relative_to(core / KERNELS).as_posix()
                       for p in (core / KERNELS).rglob("*.h"))
        cpp_files = sorted(p for p in generated.glob("*.cpp") if p.name != "compiled_preamble.cpp")
    else:
        headers = {name + ".h" for name in names}
        cpp_files = [generated / (name + ".cpp") for name in names]

    for relative in sorted(headers):
        label = "metal/" + relative
        try:
            expected = copied_text(read(input_path(KERNELS + relative)), relative)
            if read(generated / label) != expected:
                raise ValueError("copied body differs from transformed core")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append({"resource": label, "error": str(error)})

    for cpp in cpp_files:
        try:
            actual = read(cpp)
            roots = re.findall(r'^// Auto generated source for ([^\n]+)$', actual, re.MULTILINE)
            declared = re.findall(r'^// Contents from "([^"]+)"$', actual, re.MULTILINE)
            if len(roots) != 1 or not declared or declared[-1] != roots[0]:
                raise ValueError("missing/ambiguous root or final root section")
            if len(set(declared)) != len(declared):
                raise ValueError("duplicate declared section")
            if not all_sources and roots[0] != KERNELS + cpp.stem + ".h":
                raise ValueError("unexpected root for scoped resource")
            sections = [(path, read(input_path(path))) for path in declared]
            expected = jit_text(cpp.stem, roots[0], sections)
            resources.append({"resource": cpp.name, "declared_inputs": declared,
                              "expected_sha256": sha(expected.encode())})
            if actual != expected:
                raise ValueError("expanded body or wrapper differs from current core")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append({"resource": cpp.name, "error": str(error)})

    for path in (core / "mlx/backend/metal/make_compiled_preamble.sh",
                 package / "tools/fix-metal-includes.sh"):
        try:
            read(path)
        except (OSError, UnicodeError) as error:
            errors.append({"resource": str(path.relative_to(package)), "error": str(error)})
    hashes["checker"] = sha(Path(__file__).read_bytes())
    identity = "".join(f"{digest}  {path}\n" for path, digest in sorted(hashes.items()))
    try:
        revision = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"schema": 1, "scope": "all-metal" if all_sources else list(names),
            "passed": not errors, "copied_files": len(headers), "jit_files": len(cpp_files),
            "core_revision": revision, "source_identity_sha256": sha(identity.encode()),
            "hashes": hashes, "resources": resources, "errors": errors,
            "limitation": "Declared-section consistency only; compiler include discovery and compiled binary identity are not verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="Audit all Metal resources; existing drift fails")
    scope.add_argument("--resource", action="append", help="Root kernel basename; repeat to scope")
    parser.add_argument("--manifest", type=Path, help="Write a JSON source receipt, including failures")
    args = parser.parse_args()
    names = args.resource or DEFAULT_NAMES
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in names):
        parser.error("resource must be a kernel basename")
    result = audit(args.package_root, names, args.all)
    if args.manifest:
        args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"generated-metal: {'PASS' if result['passed'] else 'FAIL'}; "
          f"{result['copied_files']} copied files, {result['jit_files']} JIT resources; "
          f"source identity {result['source_identity_sha256']}")
    for error in result["errors"]:
        print(f"{error['resource']}: {error['error']}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
