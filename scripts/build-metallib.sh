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
# Usage:
#   scripts/build-metallib.sh                 Build, then copy into the test bundles.
#                                             Run it after `swift build --build-tests`.
#   scripts/build-metallib.sh build FILE      Build only, and write the library to FILE.
#   scripts/build-metallib.sh install FILE    Copy the library FILE into the test bundles.
#
# CI builds the library in one job and the tests in another job. The library
# depends only on the mlx submodule, so CI caches it and builds it again only
# when the mlx pin changes.

set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
build="$repo/build/metallib"

build_metallib() {
    cmake -S "$repo/Source/Cmlx/mlx" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLX_METAL_JIT=OFF \
        -DMLX_BUILD_TESTS=OFF \
        -DMLX_BUILD_EXAMPLES=OFF \
        -DMLX_BUILD_BENCHMARKS=OFF \
        -DMLX_BUILD_PYTHON_BINDINGS=OFF
    cmake --build "$build" --target mlx-metallib -j "$(sysctl -n hw.ncpu)"
}

install_metallib() {
    local metallib=$1
    if [ ! -f "$metallib" ]; then
        echo "The Metal library $metallib does not exist." >&2
        exit 1
    fi
    bundles=$(find "$repo/.build" -path '*.xctest/Contents/MacOS' -type d)
    if [ -z "$bundles" ]; then
        echo "No test bundle under .build. Run swift build --build-tests first." >&2
        exit 1
    fi
    while IFS= read -r dir; do
        cp "$metallib" "$dir/"
        echo "Copied mlx.metallib to $dir"
    done <<< "$bundles"
}

built="$build/mlx/backend/metal/kernels/mlx.metallib"

case "${1:-}" in
    "")
        build_metallib
        install_metallib "$built"
        ;;
    build)
        if [ $# -ne 2 ]; then
            echo "Usage: $0 build FILE" >&2
            exit 2
        fi
        build_metallib
        mkdir -p "$(dirname "$2")"
        cp "$built" "$2"
        echo "Wrote $2"
        ;;
    install)
        if [ $# -ne 2 ]; then
            echo "Usage: $0 install FILE" >&2
            exit 2
        fi
        install_metallib "$2"
        ;;
    *)
        echo "Usage: $0 [build FILE | install FILE]" >&2
        exit 2
        ;;
esac
