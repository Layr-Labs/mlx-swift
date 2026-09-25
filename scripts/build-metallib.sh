#!/bin/bash
# Build the full MLX Metal library and copy it into each test bundle.
#
# A SwiftPM command-line build compiles no Metal kernels, so `swift test`
# finds no metallib. MLX first looks for mlx.metallib next to the binary
# that links MLX. For `swift test`, that binary is inside the .xctest bundle.
#
# This script builds mlx.metallib with cmake from the mlx submodule. It
# compiles every kernel ahead of time (MLX_METAL_JIT=OFF). Then it copies
# the file into every .xctest/Contents/MacOS/ folder under .build.
#
# Run it after `swift build --build-tests`.

set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
build="$repo/build/metallib"

cmake -S "$repo/Source/Cmlx/mlx" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_METAL_JIT=OFF \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$build" --target mlx-metallib -j "$(sysctl -n hw.ncpu)"
metallib="$build/mlx/backend/metal/kernels/mlx.metallib"

bundles=$(find "$repo/.build" -path '*.xctest/Contents/MacOS' -type d)
if [ -z "$bundles" ]; then
    echo "No test bundle under .build. Run swift build --build-tests first." >&2
    exit 1
fi
while IFS= read -r dir; do
    cp "$metallib" "$dir/"
    echo "Copied mlx.metallib to $dir"
done <<< "$bundles"
