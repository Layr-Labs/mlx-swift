#!/usr/bin/env python3
"""Isolated mutation tests for the generated Metal source gate; no compiler."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check-generated-metal-sources.py")
spec = importlib.util.spec_from_file_location("generated_metal", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

# Fixed generator-format fixture, independent of the checker's reconstruction.
CPP = '''namespace mlx::core::metal {

const char* sample() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/sample.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/dependency.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/dependency.h"
constant int dependency = 7;

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/sample.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/sample.h"
#include <metal_stdlib>
constant int value = 3;

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
'''


class GeneratedMetalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.core = self.root / "Source/Cmlx/mlx/mlx/backend/metal/kernels"
        self.generated = self.root / "Source/Cmlx/mlx-generated"
        self.core.mkdir(parents=True)
        (self.generated / "metal").mkdir(parents=True)
        (self.root / "tools").mkdir()
        (self.core.parent / "make_compiled_preamble.sh").write_text("fixture generator\n")
        (self.root / "tools/fix-metal-includes.sh").write_text("fixture include transform\n")
        self.source = '#include <metal_stdlib>\n#include "mlx/backend/metal/kernels/dependency.h"\nconstant int value = 3;\n'
        (self.core / "sample.h").write_text(self.source)
        (self.core / "dependency.h").write_text("#pragma once\nconstant int dependency = 7;\n")
        (self.generated / "metal/sample.h").write_text(self.source.replace("mlx/backend/metal/kernels/", ""))
        self.cpp = self.generated / "sample.cpp"
        self.cpp.write_text(CPP)

    def audit(self):
        return checker.audit(self.root, ["sample"])

    def test_exact_fixture_and_receipt(self):
        result = self.audit()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(len(result["resources"][0]["declared_inputs"]), 2)
        self.assertEqual(result["source_identity_sha256"], self.audit()["source_identity_sha256"])
        self.assertIn("Source/Cmlx/mlx/mlx/backend/metal/kernels/dependency.h", result["hashes"])

    def test_mutated_copied_body_fails(self):
        copied = self.generated / "metal/sample.h"
        copied.write_text(copied.read_text().replace("value = 3", "value = 4"))
        self.assertFalse(self.audit()["passed"])

    def test_mutated_expanded_body_fails(self):
        self.cpp.write_text(CPP.replace("value = 3", "value = 4"))
        self.assertFalse(self.audit()["passed"])

    def test_source_without_final_newline_matches_generator_grep(self):
        (self.core / "dependency.h").write_text("#pragma once\nconstant int dependency = 7;")
        self.assertTrue(self.audit()["passed"])

    def test_mutated_dependency_fails(self):
        (self.core / "dependency.h").write_text("constant int dependency = 8;\n")
        self.assertFalse(self.audit()["passed"])

    def test_wrapper_mutation_fails(self):
        self.cpp.write_text(CPP.replace("const char* sample()", "const char* other()"))
        self.assertFalse(self.audit()["passed"])

    def test_missing_root_section_fails(self):
        self.cpp.write_text(CPP.replace('// Contents from "mlx/backend/metal/kernels/sample.h"', '// root omitted'))
        self.assertFalse(self.audit()["passed"])

    def test_absolute_declared_input_is_rejected(self):
        self.cpp.write_text(CPP.replace("mlx/backend/metal/kernels/dependency.h", "/old/toolchain/dependency.h"))
        self.assertIn("non-core declared input", self.audit()["errors"][0]["error"])

    def test_nested_copied_include_transform(self):
        source = '#include "mlx/backend/metal/kernels/dependency.h"\n#include <metal_stdlib>\n'
        self.assertEqual(checker.copied_text(source, "steel/gemm/sample.h"),
                         '#include "../../dependency.h"\n#include <metal_stdlib>\n')

    def test_all_mode_reports_missing_copy_and_unknown_jit_wrapper(self):
        (self.generated / "unknown.cpp").write_text("broken wrapper\n")
        result = checker.audit(self.root, all_sources=True)
        self.assertFalse(result["passed"])
        failures = {error["resource"] for error in result["errors"]}
        self.assertIn("metal/dependency.h", failures)
        self.assertIn("unknown.cpp", failures)

    def test_cli_failure_receipt_and_exit(self):
        self.cpp.write_text(CPP.replace("value = 3", "value = 9"))
        receipt = self.root / "receipt.json"
        result = subprocess.run([sys.executable, str(SCRIPT), "--package-root", str(self.root),
                                 "--resource", "sample", "--manifest", str(receipt)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(receipt.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
