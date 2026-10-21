#!/bin/sh
# CPU ledger and C bridge checks; does not create a Metal device.
set -eu
cd "$(dirname "$0")/.."
cap_output=$(mktemp -d "${TMPDIR:-/tmp}/mlx-cap-check.XXXXXX")
trap 'rm -rf -- "$cap_output"' EXIT HUP INT TERM
xcrun clang++ -std=c++20 -fblocks -pthread -I Source/Cmlx/mlx \
  Source/Cmlx/mlx/tests/standalone/process_allocation_cap.cpp -o "$cap_output/ledger-tests"
"$cap_output/ledger-tests"
xcrun clang++ -std=c++20 -pthread -I Source/Cmlx/mlx -I Source/Cmlx/mlx-c \
  Source/Cmlx/mlx-c/mlx/c/memory_cap.cpp \
  Source/Cmlx/mlx-c/examples/cap-tests/memory_cap.cpp -o "$cap_output/c-api-tests"
"$cap_output/c-api-tests"
xcrun clang++ -std=c++20 -fblocks -fsyntax-only -I Source/Cmlx/mlx -I Source/Cmlx/metal-cpp \
  Source/Cmlx/mlx/mlx/backend/metal/allocator.cpp
xcrun clang++ -std=c++20 -fsyntax-only -I Source/Cmlx/mlx Source/Cmlx/mlx/mlx/backend/no_gpu/allocator.cpp
xcrun clang -std=c11 -fsyntax-only -I Source/Cmlx/mlx-c -x c Source/Cmlx/mlx-c/mlx/c/memory.h
xcrun swiftc -frontend -parse Source/MLX/Memory+AllocationCap.swift
